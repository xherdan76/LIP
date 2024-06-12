"""
Load data from splishsplash
"""
import sys
sys.path.append('..')

import os
import json
import glob
import pickle as pkl
import joblib
import numpy as np
import os.path as osp

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class ParticleDataset(Dataset):
    def __init__(self, data_path, data_type, start, end, random_rot=True, window=3):
        """
        window: 3
            f0 --> f1 --> f2
        """
        super(ParticleDataset, self).__init__()
        self.random_rot = random_rot
        self.window = window
        self.root_dir = data_path
        self.start = start
        self.end = end
        if data_type == 'raw':
            self.dataitems = self.collect_particles_raw()
        elif data_type == 'blender':
            self.collect_particles_blender()
        self.read_box()
        # print('Total lens:',len(self.dataitems) )
        print('Total lens:', self.particles_poss.shape[0])

    def read_box(self):
        box_info = joblib.load(osp.join(self.root_dir, 'box.pt'))
        self.box = box_info['box']
        self.box_normals = box_info['box_normals']


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

    
    def collect_particles_blender(self):
        self.particles_poss, self.particles_vels = [], []
        for i in range(self.start, self.end):
            particle_pos, particle_vel = self._read_particles(osp.join(self.root_dir, 'particles', 'fluid_%04d.npz' % i))
            self.particles_poss.append(particle_pos)
            self.particles_vels.append(particle_vel)
        self.particles_poss = np.stack(self.particles_poss, 0)
        self.particles_vels = np.stack(self.particles_vels, 0)
        print(self.particles_poss.shape)


    def collect_particles_raw(self,):
        particles_dirs = glob.glob(osp.join(self.root_dir, 'sim*'))
        self.total_num = 0
        samples = []
        for particles_dir in particles_dirs:
            particle_paths_i = glob.glob(osp.join(particles_dir, 'output/fluid_*.npz'))
            particle_paths_i.sort(key=lambda x:int(x.split('_')[-1][:-4]))
            particle_paths_i = particle_paths_i[self.start:self.end]
            box_path = osp.join(particles_dir, 'box.pt')
            box, box_normal = self._read_box(box_path)
            for idx in range(len(particle_paths_i)-self.window):
                sample = {
                            'box': box,
                            'box_normals': box_normal,
                            }
                for ii in range(self.window):
                    _pos, _vel = self._read_particles(particle_paths_i[idx+ii])
                    sample[f'particles_pos_{ii}'] = _pos
                    sample[f'particles_vel_{ii}'] = _vel
                samples.append(sample)
        return samples


    def __getitem__(self, index):
        # data = self.dataitems[index]
        # returned_data = {}
        # if self.random_rot:
        #     angle = np.random.uniform(0, 2*np.pi)
        #     s = np.sin(angle)
        #     c = np.cos(angle)
        #     # rot z angle
        #     rand_R = np.array([c, -s, 0, s, c, 0, 0, 0, 1], dtype=np.float32).reshape((3,3))
        #     for k,v in data.items():
        #         returned_data[k] = torch.from_numpy(np.matmul(v, rand_R)).float()
        # else:
        #     for k,v in data.items():
        #         returned_data[k] = torch.from_numpy(v).float()
        # return returned_data
        data = {}
        data['box'] = torch.from_numpy(self.box).float()
        data['box_normals'] = torch.from_numpy(self.box_normals).float()
        data['particles_pos_0'] = torch.from_numpy(self.particles_poss[index]).float()
        data['particles_vel_0'] = torch.from_numpy(self.particles_vels[index]).float()
        data['particles_pos_1'] = torch.from_numpy(self.particles_poss[index+1]).float()
        data['particles_vel_1'] = torch.from_numpy(self.particles_vels[index+1]).float()
        return data
        

    def __len__(self):
        return self.particles_poss.shape[0] - 1


if __name__ == '__main__':
    dataset = ParticleDataset()
    print('Done')