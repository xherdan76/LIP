import numpy as np
from scipy.spatial import distance_matrix
from scipy.spatial import cKDTree
import json
import geomloss
import torch


emd_loss = lambda x, y: geomloss.SamplesLoss()(x, y)


def _distance(x, y):
    return np.linalg.norm(x - y, axis=-1)


def _ground_truth_to_prediction_distance(pred, gt):
    tree = cKDTree(pred)
    dist, _ = tree.query(gt)
    return dist


def _compute_stats(x):
    tmp = {
        'mean': np.mean(x),
        'mse': np.mean(x**2),
        'var': np.var(x),
        'min': np.min(x),
        'max': np.max(x),
        'median': np.median(x),
    }
    tmp = {k: float(v)*1000 for k, v in tmp.items()}
    tmp['num_particles'] = x.shape[0]
    return tmp


class FluidErrors:

    def __init__(self, log_emd=False):
        self.log_emd = log_emd
        self.errors = {}
        
    def cal_errors(self, pred_pos, gt_pos, time_idx):
        if self.log_emd:
            emd_distance = 1000 * emd_loss(pred_pos, gt_pos).item()
            
            if isinstance(pred_pos, torch.Tensor):
                pred_pos = pred_pos.detach().cpu().numpy()
                gt_pos = gt_pos.detach().cpu().numpy()
        
        if np.count_nonzero(~np.isfinite(pred_pos)):
            print('predicted_pos contains nonfinite values')
            return
        if np.count_nonzero(~np.isfinite(gt_pos)):
            print('gt_pos contains nonfinite values')
            return
        
        # errs = _compute_stats(_distance(pred_pos, gt_pos))
        errs = {}
        if self.log_emd:
            errs['emd'] = emd_distance
        
        gt_to_pred_distances = _ground_truth_to_prediction_distance(pred_pos, gt_pos)
        gt_to_pred_errs = _compute_stats(gt_to_pred_distances)
        for k, v in gt_to_pred_errs.items():
            errs['gt2pred_' + k] = v

        
                
        if not time_idx in self.errors:
            self.errors[time_idx] = errs
        else:
            self.errors[time_idx].update(errs)
        if self.log_emd:
            return errs['gt2pred_mean'], errs['emd']
        else:
            return errs['gt2pred_mean']
       

    def get_keys(self):
        scene_ids = set()
        init_frames = set()
        current_frames = set()
        for scene_id, init_frame, current_frame in self.errors:
            scene_ids.add(scene_id)
            init_frames.add(init_frame)
            current_frames.add(current_frame)
        return sorted(scene_ids), sorted(init_frames), sorted(current_frames)


    def save(self, path):
        with open(path, 'w') as f:
            tmp = list(self.errors.items())
            json.dump(tmp, f, indent=4)


    def load(self, path):
        with open(path, 'r') as f:
            tmp = json.load(f)
            self.errors = {tuple(k): v for k, v in tmp}
