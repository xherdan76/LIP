"""
Evaluate render
"""

import os
import json
import glob
import imageio
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import Dataset
from trainer.basetrainer import BaseTrainer

from utils.ray_utils import get_ray_directions, get_rays
from utils.particles_utils import read_obj
from models.renderer import RenderNet
from utils import eval_utils

to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

class ParticleDatset(Dataset):
    def __init__(self, particle_dir, start_index, end_index):
        self.particle_files = sorted(glob.glob(os.path.join(particle_dir, '*.npz')))[start_index:end_index]
        # print(self.particle_files)

    def __getitem__(self, index):
        particle_pos, _ = self._read_particles(self.particle_files[index])
        name = self.particle_files[index].split('/')[-1][:-4]
        return torch.from_numpy(particle_pos).float(), name

    def __len__(self,):
        return len(self.particle_files)

    def _read_particles(self, particle_path):
        """
        read initial particle information and the bounding box information
        """
        particle_info = np.load(particle_path)
        particle_pos = particle_info['pos']
        particle_vel = particle_info['vel']
        # import ipdb;ipdb.set_trace()
        particle_pos = particle_pos
        particle_vel = particle_vel
        return particle_pos, particle_vel


class RendererEvaluation(BaseTrainer):
    def __init__(self, options):
        self.options = options
        self.exppath = os.path.join(options.expdir, options.expname)
        os.makedirs(self.exppath, exist_ok=True)
        self.device = torch.device('cuda')

        self.renderer = RenderNet(self.options.RENDERER, near=self.options.TEST.near, far=self.options.TEST.far).to(self.device)
        if self.options.resume_from:
            ckpt = torch.load(self.options.resume_from)['renderer_state_dict']
            print(f'\n---> load pretrained renderer model: {self.options.resume_from} \n')
        elif self.options.TEST.pretained_renderer != '':
            ckpt = torch.load(self.options.TEST.pretained_renderer)['renderer_state_dict']
            print(f'\n---> load pretrained renderer model: {self.options.TEST.pretained_renderer} \n')
        render_state_dict = self.renderer.state_dict()
        render_state_dict.update(ckpt)
        self.renderer.load_state_dict(render_state_dict, strict=True)


        self.dataset = ParticleDatset(particle_dir=os.path.join(self.options.TEST.data_path, 'particles'), start_index=self.options.TEST.start_index, end_index=self.options.TEST.end_index)
        self.dataset_length = len(self.dataset)

        init_particle_path = self.options.TRAIN.init_particle_path
        if init_particle_path:
            print('---> Initial position', init_particle_path)
            self.init_pos = torch.Tensor(np.load(init_particle_path)['particles']).to(self.device)
        else:
            self.init_pos = None

    def pre_request(self):
        test_view = self.options.TEST.test_view
        print('testing:', test_view)
        data_dir = self.options.TEST.data_path
        W, H = self.options.TEST.imgW, self.options.TEST.imgH
        if self.options.TEST.scale != 1:
            W, H = int(W // self.options.TEST.scale), int(H // self.options.TEST.scale)
        data_dir = os.path.join(data_dir, test_view)
        with open(os.path.join(data_dir, f'transforms_train.json'), 'r') as f:
            meta = json.load(f)
        if 'camera_angle_x' in meta.keys():
            focal = .5 * W / np.tan(0.5 * meta['camera_angle_x'])
        else:
            if self.options.TEST.scale != 1:
                focal = meta['focal'] / self.options.TEST.scale
            else:
                focal = meta['focal']
        trans_matrix = np.array(meta['frames'][0]['transform_matrix'])[:3, :4]
        directions = get_ray_directions(H, W, focal)
        rays_o, rays_d = get_rays(directions, torch.FloatTensor(trans_matrix))
        rays = torch.cat([rays_o, rays_d], -1)
        ret = {'cw': torch.from_numpy(trans_matrix).float(),
                'focal': focal,
                'rays': rays.view(-1, 6),
                }

        all_rgbs = []
        for data_idx in range(self.options.TEST.start_index, self.options.TEST.end_index):
            image_path = os.path.join(data_dir, '{}.png'.format(meta['frames'][data_idx]['file_path']))

            image = np.array(imageio.imread(image_path)) / 255.
            # if self.half_res:
            if self.options.TEST.scale != 1:
                image = image.resize((W, H), Image.LANCZOS)
            image = image[..., :3]*image[..., -1:] + (1-image[..., -1:])
            all_rgbs.append(image)
        return ret, all_rgbs


    def visulization_single_image(self, rgbs, prefix, path=None):
        image = self.vis_rgbs(rgbs)
        rgb8 = to8b(image.permute(1,2,0).detach().numpy())
        if not path:
            filename = '{}/{}.png'.format(os.path.join(self.exppath, 'render_GT'), prefix)
        else:
            filename = '{}/{}.png'.format(path, prefix)
        imageio.imwrite(filename, rgb8)
        return rgb8


    def vis_rgbs(self, rgbs, channel=3):
        imgW = self.options.TEST.imgW
        imgH = self.options.TEST.imgH
        image = rgbs.reshape(imgH, imgW, channel).detach().cpu().numpy()
        return image


    def eval(self,):
        self.renderer.eval()
        render_params, all_rgbs = self.pre_request()
        render_params = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k,v in render_params.items()}
        cw  = render_params['cw'].to(self.device)
        focal_length = render_params['focal']
        rays = render_params['rays'].to(self.device)

        render_GT_dir = os.path.join(self.exppath, 'render_GT')
        if not os.path.exists(render_GT_dir):
            os.makedirs(render_GT_dir)
        rgbs = []
        psnrs = []
        ssims = []
        lpips_vgg = []
        with torch.no_grad():
            for data_idx in tqdm(range(self.options.TEST.start_index, self.options.TEST.end_index)):
                if data_idx > 52:
                    break
                gt_pos, name = self.dataset[data_idx]
                gt_pos = gt_pos.to(self.device)
                if data_idx == 0 and self.init_pos is not None:
                    gt_pos = self.init_pos
                ro = self.renderer.set_ro(cw)
                render_ret = self.render_image(gt_pos, rays.shape[0], ro, rays, focal_length, cw, iseval=True)
                # pred_rgbs_0 = render_ret['pred_rgbs_0']
                # self.visulization_single_image(pred_rgbs_0, prefix=f'coarse_pred_{name}')
                if self.options.RENDERER.ray.N_importance>0:
                    pred_rgbs_1 = render_ret['pred_rgbs_1']

                    image = self.vis_rgbs(pred_rgbs_1)
                    rgb8 = to8b(image)
                    filename = '{}/{}.png'.format(os.path.join(self.exppath, 'render_GT'), f'fine_pred_{name}')
                    imageio.imwrite(filename, rgb8)
                    rgbs.append(rgb8)
                p = -10. * np.log10(np.mean(np.square(image - all_rgbs[data_idx])))
                psnrs.append(p)
                ssims.append(eval_utils.rgb_ssim(image, all_rgbs[data_idx], max_val=1))
                # lpips_alex.append(eval_utils.rgb_lpips(rgb, all_rgbs[data_idx], net_name='alex', device=self.device))
                lpips_vgg.append(eval_utils.rgb_lpips(all_rgbs[data_idx].astype('float32'), image.astype('float32'), net_name='vgg', device=self.device))

        if len(psnrs):
            print('Testing psnr', np.mean(psnrs), '(avg)')
            print('Testing ssim', np.mean(ssims), '(avg)')
            print('Testing lpips (vgg)', np.mean(lpips_vgg), '(avg)')

        rgbs = np.array(rgbs)
        imageio.mimwrite(os.path.join(self.exppath,f'video_fine.rgb.mp4'), rgbs, fps=24, quality=8)

        with open(os.path.join(self.exppath, 'mean.json'), 'w') as f:
            info = {}
            info['avg_psnrs'] = np.mean(psnrs)
            info['avg_ssims'] = np.mean(ssims)
            info['avg_lpips (vgg)'] = np.mean(lpips_vgg)

            info['psnrs'] = psnrs
            info['ssims'] = ssims
            info['lpips (vgg)'] = lpips_vgg
            json.dump(info, f, indent=4)

            # print('Testing lpips (alex)', np.mean(lpips_alex), '(avg)')

            # pred_files = sorted(glob.glob(os.path.join('119999', 'pred_*.obj')))
            # for file in tqdm(pred_files):
            #     pred_pos = read_obj(file)
            #     pred_pos = torch.Tensor(pred_pos).to(self.device)

            #     name = file.split('/')[-1][5:-4]
            #     ro = self.renderer.set_ro(cw)
            #     render_ret = self.render_image(pred_pos, rays.shape[0], ro, rays, focal_length, cw, iseval=True)
            #     pred_rgbs_0 = render_ret['pred_rgbs_0']
            #     self.visulization_single_image(pred_rgbs_0, prefix=f'coarse_pred_{name}', path=render_predpos_dir)
            #     if self.options.RENDERER.ray.N_importance>0:
            #         pred_rgbs_1 = render_ret['pred_rgbs_1']
            #         self.visulization_single_image(pred_rgbs_1, prefix=f'fine_pred_{name}', path=render_predpos_dir)


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    from configs import warmup_training_config
    cfg = warmup_training_config()
    evaluator = RendererEvaluation(cfg)
    evaluator.eval()
