"""
The whole framework
"""

import json
import os
import os.path as osp
import random

import numpy as np
import torch
import torch.nn as nn
import utils.utils as utils
from datasets.dataset import BlenderDataset
from models.encoder import GaussianGRU, ParticleEncoder
from models.renderer import RenderNet
from models.transmodel import ParticleNet
from torch import distributions as torchd
from tqdm import tqdm
from utils.particles_utils import record2obj
from utils.point_eval import FluidErrors

from .basetrainer import BaseTrainer

img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]).cuda())
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

class Trainer(BaseTrainer):
    def init_fn(self):
        self.start_step = 0
        self.eval_count = 0
        self.encoder_dim = self.options['encoder']['stoch']
        latent_dim = self.encoder_dim * 2 if 'sample' in self.options.TRAIN.get_feat else self.encoder_dim
        if self.options.TRAIN.get_feat == 'particle_sample_multi':
            latent_dim = [self.options.TRAIN.particle_res, latent_dim]
        self.latent = torch.randn(latent_dim)
        self.build_dataloader()
        if self.options.TRAIN.LR.latent_lr != 0:
            self.current_stage = 1
        elif self.options.TRAIN.LR.latent_lr == 0 and self.options.TRAIN.LR.encoder_lr != 0:
            self.current_stage = 2
        self.build_model()
        if self.options.TRAIN.LR.latent_lr != 0:
            self.latent = nn.Parameter(self.latent)
        if self.options.TRAIN.LR.latent_lr != 0:
            self.build_latent_optimizer()
            print('\033[1;35m Current stage: stage B\033[0m')
            self.current_stage = 1
        elif self.options.TRAIN.LR.latent_lr == 0 and self.options.TRAIN.LR.encoder_lr != 0:
            self.build_encoder_optimizer()
            print('\033[1;35m Current stage: stage C\033[0m')
            self.current_stage = 2
        self.set_RGB_criterion()
        self.set_L1_criterion()
        self.save_interval = self.options.TRAIN.save_interval
        self.log_interval = self.options.TRAIN.log_interval
        if self.options.TRAIN.get_feat == 'particle_sample_multi':
            self.feat_fn = self.get_feat_multi
        else:
            raise ValueError

        self._discrete = False

        if self.options.TRAIN.LR.renderer_lr != 0. or self.options.TRAIN.LR.trans_lr != 0.:
            self.build_optimizer()
        else:
            self.transition_model.requires_grad_(False)
            self.renderer.requires_grad_(False)

        init_particle_path = self.options.TRAIN.init_particle_path
        if init_particle_path:
            print('---> Initial position', init_particle_path)
            try:
                self.init_pos = torch.Tensor(np.load(init_particle_path)['particles']).to(self.device)
            except:
                self.init_pos = torch.Tensor(np.load(init_particle_path)['pos']).to(self.device)
            if self.options.TRAIN.particle_res < self.init_pos.shape[0]:
                rand_idx = np.random.permutation(self.init_pos.shape[0])[:self.options.TRAIN.particle_res]
                self.init_pos = self.init_pos[rand_idx]
        else:
            self.init_pos = None

        self.best_gt2pred = np.inf


    def build_dataloader(self):
        self.train_view_names = self.options['train'].views.dynamic
        self.test_viewnames = self.options['test'].views
        self.dataset = BlenderDataset(self.options.train.path, self.options,
                                            imgW=self.options.TRAIN.imgW, imgH=self.options.TRAIN.imgH,
                                            imgscale=self.options.TRAIN.scale, viewnames=self.train_view_names, split='train')
        self.dataset_length = len(self.dataset)
        self.test_dataset = BlenderDataset(self.options.train.path, self.options,
                                            imgW=self.options.TEST.imgW, imgH=self.options.TEST.imgH,
                                            imgscale=self.options.TEST.scale, viewnames=self.test_viewnames, split='train')
        self.test_dataset_length = len(self.test_dataset)
        print('---> dataloader has been build')


    def build_model(self):
        # build model
        gravity = self.options.gravity
        print('---> set gravity', gravity)
        self.transition_model = ParticleNet(gravity=gravity, other_feats_channels=self.encoder_dim).to(self.device)
        self.renderer = RenderNet(self.options.RENDERER, near=self.options.near, far=self.options.far).to(self.device)

        # load pretrained checkpoints
        if self.options.TRAIN.pretrained_transition_model != '':
            self.load_pretained_transition_model(self.options.TRAIN.pretrained_transition_model)
            # self.load_pretained_encoder_model(self.options.TRAIN.pretrained_transition_model)
            print('\033[1;35m load: \033[0m', self.options.TRAIN.pretrained_transition_model)
        if self.options.TRAIN.pretained_renderer != '':
            self.load_pretained_renderer_model(self.options.TRAIN.pretained_renderer, partial_load=self.options.TRAIN.partial_load)
            print('\033[1;35m load: \033[0m', self.options.TRAIN.pretained_renderer)

        if self.options.TRAIN.use_encoder:
            if self.options['encoder']['input_last_latent']:
                encoder_dim = self.options['encoder']['stoch']
                if self.options['encoder'].get('use_std', False):
                    encoder_dim *= 2
                self.encoder = ParticleEncoder(other_feats_channels=encoder_dim).to(self.device)
            else:
                self.encoder = ParticleEncoder().to(self.device)
            self.prior_gru = GaussianGRU(output_size=self.options['encoder']['stoch'], mean_act=self.options['encoder']['mean_act']).to(self.device)
            checkpoint = torch.load(self.options.TRAIN.pretrained_transition_model)
            self.encoder.load_state_dict(checkpoint['encoder'], strict=True)
            print('\n load encoder encoder:')
            self.prior_gru.load_state_dict(checkpoint['prior_gru'], strict=True)
            print('\n load encoder gru:')
            if 'latent' in checkpoint.keys():
                self.latent = checkpoint['latent']
                print('\n loaded latent')

        if self.options.TRAIN.pretrained_latent != '':
            checkpoint = torch.load(self.options.TRAIN.pretrained_latent)
            self.latent = checkpoint['latent']
            print('\n loaded latent')
        else:
            if self.current_stage > 1:
                try:
                    latent_path = os.path.join(self.exppath, '../../stageB/latent1e-4/models/98000.pt')
                    checkpoint = torch.load(latent_path)
                    self.latent = checkpoint['latent']
                    print('\n loaded latent')
                    print('loaded stage B last checkpoint:\n --->', latent_path)
                except:
                    raise NotImplementedError


    def build_optimizer(self):
        if self.options.TRAIN.loss_weight['encoder_KL_loss'] != 0.:
            self.encoder.requires_grad_(False)
        renderer_lr = self.options.TRAIN.LR.renderer_lr
        transition_lr = self.options.TRAIN.LR.trans_lr
        seperate_render_transition = self.options.TRAIN.seperate_render_transition
        if seperate_render_transition:
            self.optimizer = torch.optim.Adam([
                {'params': self.renderer.parameters(), 'lr': renderer_lr},
            ])
            self.transition_optimizer = torch.optim.Adam([
                {'params': self.transition_model.parameters(), 'lr': transition_lr},
            ])
        else:
            self.optimizer = torch.optim.Adam([
                {'params': self.renderer.parameters(), 'lr': renderer_lr},
                {'params': self.transition_model.parameters(), 'lr': transition_lr}
                ])
        if self.options.TRAIN.LR.use_scheduler:
            boundaries = [
                50000,  # 10k
                100000,  # 75k
                200000,  # 150k
            ]
            lr_values = [
                1.0,
                0.5,
                0.25,
                0.125,
            ]

            def lrfactor_fn(x):
                factor = lr_values[0]
                for b, v in zip(boundaries, lr_values[1:]):
                    if x > b:
                        factor = v
                    else:
                        break
                return factor

            self.optim_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lrfactor_fn)

            if seperate_render_transition:
                boundaries_trans = [
                    80000, # 10k
                    120000,
                    160000,
                    200000,
                    300000,
                ]
                lr_values_trans = [
                    1.0,
                    0.5,
                    0.25,
                    0.125,
                    0.5 * 0.125,
                    0.25 * 0.125,
                    0.125 * 0.125,
                ]

                def lrfactor_fn_transition(x):
                    factor = lr_values[0]
                    for b, v in zip(boundaries_trans, lr_values_trans[1:]):
                        if x > b:
                            factor = v
                        else:
                            break
                    return factor

                self.optim_lr_scheduler_transition = torch.optim.lr_scheduler.LambdaLR(self.transition_optimizer, lrfactor_fn_transition)


    def build_latent_optimizer(self):
        latent_lr = self.options.TRAIN.LR.latent_lr
        self.latent_optimizer = torch.optim.Adam([
            {'params': self.latent, 'lr': latent_lr}
        ])
        if self.options.TRAIN.LR.use_scheduler_latent:
            self.optim_lr_scheduler_latent = torch.optim.lr_scheduler.CosineAnnealingLR(self.latent_optimizer, T_max=5000, eta_min=latent_lr*0.1)


    def build_encoder_optimizer(self):
        encoder_lr = self.options.TRAIN.LR.encoder_lr
        self.encoder_optimizer = torch.optim.Adam([
            {'params': self.encoder.parameters(), 'lr': encoder_lr},
            {'params': self.prior_gru.parameters(), 'lr': encoder_lr}
        ])
        if self.options.TRAIN.LR.use_scheduler_encoder:
            self.optim_lr_scheduler_encoder = torch.optim.lr_scheduler.CosineAnnealingLR(self.encoder_optimizer, T_max=5000, eta_min=encoder_lr*0.1)


    def resume(self, ckpt_file):
        checkpoint = torch.load(ckpt_file)
        self.start_step = checkpoint['step']
        self.renderer.load_state_dict(checkpoint['renderer_state_dict'], strict=True)
        self.transition_model.load_state_dict(checkpoint['transition_model_state_dict'], strict=True)
        self.latent = nn.Parameter(checkpoint['latent'])
        latent_lr = self.options.TRAIN.LR.latent_lr
        self.latent_optimizer = torch.optim.Adam([
            {'params': self.latent, 'lr': latent_lr}
        ])

        if self.options.TRAIN.LR.use_scheduler_latent:
            self.optim_lr_scheduler_latent = torch.optim.lr_scheduler.CosineAnnealingLR(self.latent_optimizer, T_max=3000, eta_min=1e-4)
            print('\n \033[1;35m-----!!!Resume and reload latent!!!------\033[0m')
            print('start step:', self.start_step, end='\n')


    def save_checkpoint(self, global_step, is_best=False):
        if self.options.TRAIN.LR.latent_lr != 0:
            model_dicts = {'step':global_step,
                            'renderer_state_dict':self.renderer.state_dict(),
                            'transition_model_state_dict':self.transition_model.state_dict(),
                            'latent': self.latent,
                            'latent_optimizer_state_dict': self.latent_optimizer.state_dict()
                            }
        # elif self.options.TRAIN.LR.latent_lr == 0 and self.options.TRAIN.LR.encoder_lr != 0:
        else:
            model_dicts = {'step':global_step,
                            'renderer_state_dict':self.renderer.state_dict(),
                            'transition_model_state_dict':self.transition_model.state_dict(),
                            'latent': self.latent,
                            'encoder': self.encoder.state_dict(),
                            'prior_gru': self.prior_gru.state_dict(),
                            }
        torch.save(model_dicts,
                    osp.join(self.exppath, 'models', f'{global_step}.pt'))
        if is_best:
            torch.save(model_dicts,
                    osp.join(self.exppath, 'models', f'best.pt'))


    def train(self,):
        # prepare training
        global_step = self.start_step
        if self.options.TRAIN.epochs != 0:
            self.eval(global_step)
        view_num = len(self.train_view_names)
        imgW, imgH = self.options.TRAIN.imgW, self.options.TRAIN.imgH
        img_scale = self.options.TRAIN.scale
        H = int(imgH // img_scale)
        W = int(imgW // img_scale)

        # self.transition_model.eval()
        self.renderer.eval()

        for epoch_idx in tqdm(range(int(self.start_step / 49), self.options.TRAIN.epochs), total=self.options.TRAIN.epochs, desc='Epoch:'):
            self.tmp_fluid_error = FluidErrors(log_emd=False)
            for data_idx in range(self.dataset_length):
                data = self.dataset[data_idx]
                keys = ['particles_vel', 'particles_pos_1', 'cw_1', 'rgb_1', 'rays_1', 'focal', 'box', 'box_normals']
                data = {k: data[k].to(self.device) if isinstance(data[k], torch.Tensor) else data[k] for k in keys}

                # training
                loss = self.train_step(data, data_idx, view_num, H, W, global_step)
                assert self.options.TRAIN.LR.renderer_lr == 0. and self.options.TRAIN.LR.trans_lr == 0
                if self.options.TRAIN.LR.latent_lr != 0:
                    self.update_step_latents(loss, global_step)
                    if global_step == 0:
                        print('latent optimizing')
                elif self.options.TRAIN.LR.latent_lr == 0 and self.options.TRAIN.LR.encoder_lr != 0:
                    self.update_encoder(loss, global_step) # train encoder only
                    if global_step == 0:
                        print('encoder optimizing')
                else:
                    raise ValueError
                global_step += 1

                # evaluation
                if global_step != 0 and global_step % self.save_interval == 0:
                    self.eval(global_step)
                    self.save_checkpoint(global_step)

        self.save_checkpoint(global_step)


    def load_init_pos(self, data):
        self.vel_for_next_step = data['particles_vel']
        # load initial particles extracted by static DVGO
        if self.init_pos is not None:
            self.pos_for_next_step = self.init_pos
        else:
            self.pos_for_next_step = data['particles_pos']


    def second_trainsition_step_for_training(self, data, data_idx):
        box = data['box']
        box_normals = data['box_normals']
        if data_idx == 0:
            self.load_init_pos(data)
            self.prior_gru.init_hidden(self.options.TRAIN.particle_res)
        self.prior_gru.stop_gradient()
        if self.options['encoder']['input_last_latent']:
            if data_idx == 0:
                if self.options['encoder'].get('use_std', False):
                    particle_feat = torch.zeros([self.options.TRAIN.particle_res, 2 * self.encoder_dim]).to(box.device)
                else:
                    particle_feat = torch.zeros([self.options.TRAIN.particle_res, self.encoder_dim]).to(box.device)
            else:
                if self.options['encoder']['use_mean']:
                    particle_feat = self.prior_feat_mean
                    if self.options['encoder'].get('use_std', False):
                        particle_feat = torch.cat([self.prior_feat_mean, self.prior_feat_std], dim=-1)
        else:
            particle_feat = None
        if particle_feat is not None:
            particle_feat = particle_feat.detach().clone()
            particle_feat.requires_grad = False
        input_prior = [self.pos_for_next_step, self.vel_for_next_step, particle_feat, box, box_normals]
        h = self.encoder(input_prior)
        particle_feat, prior_stat = self.prior_gru(h)
        self.prior_feat_mean = prior_stat['mean']
        self.prior_feat_std = prior_stat['std']
        pred_pos, pred_vel, num_fluid_nn = self.transition_model(self.pos_for_next_step, self.vel_for_next_step, box, box_normals, feats=particle_feat)

        in_mask_proportion = 1
        if self.options.TRAIN.get('outside_clip', False):
            in_mask = (pred_pos > self.pos_min).all(dim=-1) & (pred_pos < self.pos_max).all(dim=-1)
            pred_pos = torch.where(pred_pos > self.pos_max, self.pos_max, pred_pos)
            pred_pos = torch.where(pred_pos < self.pos_min, self.pos_min, pred_pos)
            pred_vel = (pred_pos - self.pos_for_next_step) / self.transition_model.time_step
            in_mask_num = in_mask.sum(dim=-1)
            in_mask_proportion = in_mask_num / pred_pos.shape[0]

        kl_loss = 0.
        if self.options.TRAIN.loss_weight['encoder_KL_loss'] != 0.:
            dist = lambda x: self.prior_gru.get_dist(x)
            kl_loss = torchd.kl.kl_divergence(dist(prior_stat)._dist, self.get_dist(self.latent)._dist)
            kl_loss = kl_loss.mean()

        self.pos_for_next_step, self.vel_for_next_step = pred_pos.clone().detach(),pred_vel.clone().detach()
        self.pos_for_next_step.requires_grad = False
        self.vel_for_next_step.requires_grad = False
        return pred_pos, kl_loss, in_mask_proportion


    def first_trainsition_step_for_training(self, data, data_idx):
        box = data['box']
        box_normals = data['box_normals']
        if data_idx == 0:
            self.load_init_pos(data)
        fluid_feats = self.feat_fn(self.pos_for_next_step)
        pred_pos, pred_vel, num_fluid_nn = self.transition_model(self.pos_for_next_step, self.vel_for_next_step, box, box_normals, feats=fluid_feats)

        in_mask_proportion = 1
        if self.options.TRAIN.get('outside_clip', False):
            in_mask = (pred_pos > self.pos_min).all(dim=-1) & (pred_pos < self.pos_max).all(dim=-1)
            pred_pos = torch.where(pred_pos > self.pos_max, self.pos_max, pred_pos)
            pred_pos = torch.where(pred_pos < self.pos_min, self.pos_min, pred_pos)
            pred_vel = (pred_pos - self.pos_for_next_step) / self.transition_model.time_step
            in_mask_num = in_mask.sum(dim=-1)
            in_mask_proportion = in_mask_num / pred_pos.shape[0]

        kl_loss = 0.
        self.pos_for_next_step, self.vel_for_next_step = pred_pos.clone().detach(),pred_vel.clone().detach()
        self.pos_for_next_step.requires_grad = False
        self.vel_for_next_step.requires_grad = False
        return pred_pos, kl_loss, in_mask_proportion


    def train_step(self, data, data_idx, view_num, H, W, global_step):
        # -----
        # particle transition
        # -----
        if self.options.TRAIN.use_latent:
            pred_pos, kl_loss, in_mask_proportion = self.first_trainsition_step_for_training(data, data_idx)
        elif not self.options.TRAIN.use_latent and self.options.TRAIN.use_encoder:
            pred_pos, kl_loss, in_mask_proportion = self.second_trainsition_step_for_training(data, data_idx)
        if global_step % self.log_interval == 0 and global_step != 0:
            pos_t1 = data['particles_pos_1']
            dist_gt2pred = self.tmp_fluid_error.cal_errors(pred_pos.detach().cpu().numpy(), pos_t1.detach().cpu().numpy(), data_idx+1)
            self.summary_writer.add_scalar(f'Train/pred2gt_distance', dist_gt2pred, global_step)
            if self.options.TRAIN.get('outside_clip', False):
               self.summary_writer.add_scalar(f'Train/in_mask_proportion', in_mask_proportion, global_step)

        # -----
        # rendering
        # -----
        ray_chunk = self.options.RENDERER.ray.ray_chunk
        N_importance = self.options.RENDERER.ray.N_importance
        total_loss = 0.
        # for view_idx in range(view_num):
        view_idx = random.choice(list(range(view_num)))
        # -------
        # render by a nerf model, and then calculate mse loss
        # -------
        view_name = self.train_view_names[view_idx]
        cw_t1 = data['cw_1'][view_idx]
        rgbs_t1 = data['rgb_1'][view_idx]
        focal_length = data['focal'][view_idx]
        rays_t1 = data['rays_1'][view_idx]
        # randomly sample pixel
        coords = self.random_sample_coords(H,W,global_step)
        coords = torch.reshape(coords, [-1,2])
        select_inds = np.random.choice(coords.shape[0], size=[ray_chunk], replace=False)
        select_coords = coords[select_inds].long()
        rays_t1 = rays_t1[select_coords[:, 0], select_coords[:, 1]]
        rgbs_t1 = rgbs_t1.view(H, W, -1)[select_coords[:, 0], select_coords[:, 1]]
        ro_t1 = self.renderer.set_ro(cw_t1)
        render_ret = self.render_image(pred_pos, ray_chunk, ro_t1, rays_t1, focal_length, cw_t1)
        # calculate mse loss
        rgbloss_0 = self.rgb_criterion(render_ret['pred_rgbs_0'], rgbs_t1[:ray_chunk])
        if N_importance>0:
            rgbloss_1 = self.rgb_criterion(render_ret['pred_rgbs_1'], rgbs_t1[:ray_chunk])
            rgbloss = rgbloss_0 + rgbloss_1
        else:
            rgbloss = rgbloss_0
        total_loss = total_loss+rgbloss

        # log
        if global_step % self.log_interval == 0 and global_step != 0:
            self.summary_writer.add_scalar(f'{view_name}/rgbloss_0', rgbloss_0.item(), global_step)
            self.summary_writer.add_scalar(f'{view_name}/rgbloss', rgbloss.item(), global_step)
            self.summary_writer.add_histogram(f'{view_name}/num_neighbors_0', render_ret['num_nn_0'], global_step)
            if N_importance>0:
                self.summary_writer.add_scalar(f'{view_name}/rgbloss_1', rgbloss_1.item(), global_step)
                self.summary_writer.add_histogram(f'{view_name}/num_neighbors_1', render_ret['num_nn_1'], global_step)

        if self.options.TRAIN.loss_weight['boundary_loss'] != 0.:
            bd_loss = self.cal_boundary_loss(pred_pos)
            total_loss = total_loss + bd_loss * self.options.TRAIN.loss_weight['boundary_loss']
            if (global_step+1) % self.log_interval == 0:
                self.summary_writer.add_scalar(f'boudary_loss', bd_loss.item(), global_step)
        if self.options.TRAIN.loss_weight['encoder_KL_loss'] != 0.:
            total_loss = total_loss + kl_loss * self.options.TRAIN.loss_weight['encoder_KL_loss']
            if (global_step+1) % self.log_interval == 0:
                self.summary_writer.add_scalar(f'encoder_kl_loss', kl_loss.item(), global_step)
        return total_loss


    def update_step_latents(self,loss, global_step):
        grad_clip_value = self.options.TRAIN.grad_clip_value

        self.latent_optimizer.zero_grad()
        loss.backward()
        if grad_clip_value != 0:
            torch.nn.utils.clip_grad_norm_(self.latent, grad_clip_value)
            self.summary_writer.add_histogram('latent_grad', self.latent.grad.norm(), global_step)
        self.latent_optimizer.step()
        if self.options.TRAIN.LR.use_scheduler_latent:
            self.optim_lr_scheduler_latent.step()

        if global_step != 0 and global_step % self.log_interval == 0:
            lrs = self.get_learning_rate(self.latent_optimizer)
            for i,lr in enumerate(lrs):
                self.summary_writer.add_scalar(f'learning_rate/lr_latent_{i}', lr, global_step)


    def update_encoder(self,loss, global_step):
        grad_clip_value = self.options.TRAIN.grad_clip_value

        self.encoder_optimizer.zero_grad()
        loss.backward()
        if grad_clip_value != 0:
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), grad_clip_value)
            encoder_grad = self.cal_grad_norm(self.encoder)
            self.summary_writer.add_histogram('encoder_grad', encoder_grad, global_step)
        self.encoder_optimizer.step()
        if self.options.TRAIN.LR.use_scheduler_encoder:
            self.optim_lr_scheduler_encoder.step()

        if global_step != 0 and global_step % self.log_interval == 0:
            lrs = self.get_learning_rate(self.encoder_optimizer)
            for i,lr in enumerate(lrs):
                self.summary_writer.add_scalar(f'learning_rate/lr_encoder_{i}', lr, global_step)


    def eval(self, step_idx):
        """
        visulize the point cloud resutls, and the image
        """
        # print('\nStep {} Eval:'.format(step_idx))
        self.eval_count += 1
        self.transition_model.eval()
        self.renderer.eval()
        if self.options.TRAIN.LR.latent_lr == 0 and self.options.TRAIN.LR.encoder_lr != 0:
            self.encoder.eval()
            self.prior_gru.eval()
        view_num = len(self.test_viewnames)
        N_importance = self.options.RENDERER.ray.N_importance
        with torch.no_grad():
            dist_pred2gt_all = []
            fluid_error = FluidErrors(log_emd=False)
            for data_idx in tqdm(range(self.test_dataset_length), desc='Step {} Eval:'.format(step_idx)):
                data = self.test_dataset[data_idx]
                keys = ['box', 'box_normals', 'particles_pos', 'particles_vel', 'particles_pos_1', 'cw_1', 'rgb_1', 'rays_1', 'focal']
                data = {k: data[k].to(self.device) if isinstance(data[k], torch.Tensor) else data[k] for k in keys}

                box = data['box']
                box_normals = data['box_normals']
                if data_idx ==0:
                    if not self.options.TRAIN.use_latent and self.options.TRAIN.use_encoder:
                        self.prior_gru.init_hidden(self.options.TRAIN.particle_res)
                    pos_for_next_step, vel_for_next_step = data['particles_pos'],data['particles_vel']
                    # dist_pred2gt, dist_emd = fluid_error.cal_errors(self.init_pos, pos_for_next_step, data_idx+1)
                    dist_pred2gt = fluid_error.cal_errors(self.init_pos.detach().cpu().numpy(), pos_for_next_step.detach().cpu().numpy(), data_idx+1)
                    dist_pred2gt_all.append(dist_pred2gt)\
                    # load initial particles extracted by static DVGO
                    if self.init_pos is not None:
                        pos_for_next_step = self.init_pos
                        # here vel equals 0
                        if self.options.TRAIN.particle_res != vel_for_next_step.shape[0]:
                            vel_for_next_step = torch.zeros_like(pos_for_next_step)
                    # fluid_feats = self.latent.repeat(pos_for_next_step.shape[0], 1)
                    if self.options.TRAIN.use_latent:
                        particle_feat = self.feat_fn(pos_for_next_step)
                if not self.options.TRAIN.use_latent and self.options.TRAIN.use_encoder:
                    if self.options['encoder']['input_last_latent']:
                        if data_idx == 0:
                            if self.options['encoder'].get('use_std', False):
                                particle_feat = torch.zeros([self.options.TRAIN.particle_res, 2 * self.encoder_dim]).to(box.device)
                            else:
                                particle_feat = torch.zeros([self.options.TRAIN.particle_res, self.encoder_dim]).to(box.device)
                        else:
                            if self.options['encoder']['use_mean']:
                                particle_feat = self.prior_feat_mean
                                if self.options['encoder'].get('use_std', False):
                                    particle_feat = torch.cat([self.prior_feat_mean, self.prior_feat_std], dim=-1)
                    else:
                        particle_feat = None
                    input_prior = [pos_for_next_step, vel_for_next_step, particle_feat, box, box_normals]
                    h = self.encoder(input_prior)
                    particle_feat, prior_stat = self.prior_gru(h)
                    self.prior_feat_mean = prior_stat['mean']
                    self.prior_feat_std = prior_stat['std']

                pred_pos, pred_vel, num_fluid_nn = self.transition_model(pos_for_next_step, vel_for_next_step, box, box_normals, feats=particle_feat)
                in_mask_proportion = 1
                if self.options.TRAIN.get('outside_clip', False):
                    in_mask = (pred_pos > self.pos_min).all(dim=-1) & (pred_pos < self.pos_max).all(dim=-1)
                    pred_pos = torch.where(pred_pos > self.pos_max, self.pos_max, pred_pos)
                    pred_pos = torch.where(pred_pos < self.pos_min, self.pos_min, pred_pos)
                    pred_vel = (pred_pos - pos_for_next_step) / self.transition_model.time_step
                    in_mask_num = in_mask.sum(dim=-1)
                    in_mask_proportion = in_mask_num / pred_pos.shape[0]

                pos_for_next_step, vel_for_next_step = pred_pos.clone(), pred_vel.clone()

                # evaluate transition model
                pos_t1 = data['particles_pos_1']
                # eval pred2gt distance
                # dist_pred2gt, dist_emd = fluid_error.cal_errors(pred_pos, pos_t1, data_idx+1)
                dist_pred2gt = fluid_error.cal_errors(pred_pos.detach().cpu().numpy(), pos_t1.detach().cpu().numpy(), data_idx+1)
                dist_pred2gt_all.append(dist_pred2gt)
                # dist_emd_all.append(dist_emd)
                self.summary_writer.add_scalar(f'pred2gt_distance', dist_pred2gt, self.eval_count*self.test_dataset_length+data_idx+1)
                # save to obj
                if (step_idx / self.save_interval) % 5 == 0:
                    if not osp.exists(osp.join(self.particlepath, f'{step_idx}')):
                        os.makedirs(osp.join(self.particlepath, f'{step_idx}'))
                    particle_name = osp.join(self.particlepath, f'{step_idx}/pred_{data_idx+1}.obj')
                    with open(particle_name, 'w') as fp:
                        record2obj(pred_pos, fp, color=[255, 0, 0]) # red
                    particle_name = osp.join(self.particlepath, f'{step_idx}/gt_{data_idx+1}.obj')
                    with open(particle_name, 'w') as fp:
                        record2obj(pos_t1, fp, color=[3, 168, 158])

                # rendering results
                # to save time, we only render several frames
                if (step_idx / self.save_interval) % 20 == 0:
                # if False:
                    if data_idx in [20,30]:
                        for view_idx in range(view_num):
                            view_name = self.test_viewnames[view_idx]
                            cw = data['cw_1'][view_idx]
                            ro = self.renderer.set_ro(cw)
                            focal_length = data['focal'][view_idx]
                            rgbs = data['rgb_1'][view_idx]
                            rays = data['rays_1'][view_idx].view(-1, 6)
                            render_ret = self.render_image(pred_pos, rays.shape[0], ro, rays, focal_length, cw, iseval=True)
                            pred_rgbs_0 = render_ret['pred_rgbs_0']
                            mask_0 = render_ret['mask_0']
                            psnr_0 = mse2psnr(img2mse(pred_rgbs_0, rgbs.detach().cpu()))
                            self.summary_writer.add_scalar(f'{view_name}/psnr_{data_idx}_0', psnr_0.item(), step_idx)
                            self.visualization(pred_rgbs_0, rgbs, step_idx, mask=mask_0, prefix=f'coarse_{data_idx}_{view_name}')
                            if N_importance>0:
                                pred_rgbs_1 = render_ret['pred_rgbs_1']
                                mask_1 = render_ret['mask_1']
                                psnr_1 = mse2psnr(img2mse(pred_rgbs_1, rgbs.detach().cpu()))
                                self.summary_writer.add_scalar(f'{view_name}/psnr_{data_idx}_1', psnr_1.item(), step_idx)
                                self.visualization(pred_rgbs_1, rgbs, step_idx, mask=mask_1, prefix=f'fine_{data_idx}_{view_name}')
            fluid_error.save(osp.join(self.particlepath, f'res_{step_idx}.json'))
            path = osp.join(self.exppath, f'avg_pred2gt.json')
            mean_pred2gt = np.mean(dist_pred2gt_all)
            self.summary_writer.add_scalar('avg_pred2gt_distance/avg_pred2gt_distance', mean_pred2gt, step_idx)
            self.summary_writer.add_scalar('avg_pred2gt_distance/avg_pred2gt_distance_0-49', np.mean(dist_pred2gt_all[:49]), step_idx)
            self.summary_writer.add_scalar('avg_pred2gt_distance/avg_pred2gt_distance_49', np.mean(dist_pred2gt_all[-1]), step_idx)

            if self.current_stage == 1:
                if mean_pred2gt< self.best_gt2pred:
                    self.best_gt2pred = mean_pred2gt
                    self.save_checkpoint(step_idx, is_best=True)

        self.transition_model.train()
        self.renderer.train()
        if self.options.TRAIN.LR.latent_lr == 0 and self.options.TRAIN.LR.encoder_lr != 0:
            self.encoder.train()
            self.prior_gru.train()


    def get_feat_multi(self, pos):
        num_particles = pos.shape[0]
        stoch_latent = self.get_dist(self.latent).sample().reshape(num_particles, -1)
        return stoch_latent


    def get_dist(self, latent, dtype=None):
        if self._discrete:
            logit = latent
            dist = torchd.independent.Independent(utils.OneHotDist(logit), 1)
        else:
            mean, std = latent.chunk(2, dim=-1)
            mean = {
                'none': lambda: mean,
                'tanh5': lambda: 5.0 * torch.tanh(mean / 5.0),
                'tanh': lambda: 1.0 * torch.tanh(mean),
            }[self.options['encoder']['mean_act']]()
            std_act = lambda std: 2 * torch.sigmoid(std / 2)
            std = std_act(std=std)
            dist = utils.ContDist(torchd.independent.Independent(torchd.normal.Normal(mean, std), 1))
        return dist