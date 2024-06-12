import torch
import torch.nn.functional as F
from torch import distributions as torchd


def interpolation(grid, xyz, xyz_min, xyz_max):
    shape = xyz.shape[:-1]
    xyz = xyz.reshape(1, 1, 1, -1, 3)
    ind_norm = ((xyz - xyz_min) / (xyz_max - xyz_min)).flip((-1, )) * 2 - 1  # N, D, H, W, 3
    out = F.grid_sample(grid, ind_norm, mode='bilinear', align_corners=True)
    out = out.reshape(grid.shape[1], -1).T.reshape(*shape, grid.shape[1])
    if grid.shape[1] == 1:
        out = out.squeeze(-1)
    return out


def norm_feat(fluid_feat, min_vis=0.01, max_vis=0.4, min_dens=500, max_dens=2000, particle_radius=0.025):
    volume = (2 * particle_radius) ** 3
    min_mass = volume * min_dens
    max_mass = volume * max_dens
    min_feat = torch.tensor([min_vis, min_mass]).to(fluid_feat.device)
    max_feat = torch.tensor([max_vis, max_mass]).to(fluid_feat.device)

    # normalize to range [-1, 1]
    fluid_feat = (fluid_feat - min_feat) / (max_feat - min_feat) * 2 -1
    return fluid_feat


class SampleDist:

    def __init__(self, dist, samples=100):
        self._dist = dist
        self._samples = samples

    @property
    def name(self):
        return 'SampleDist'

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def mean(self):
        samples = self._dist.sample(self._samples)
        return torch.mean(samples, 0)

    def mode(self):
        sample = self._dist.sample(self._samples)
        logprob = self._dist.log_prob(sample)
        return sample[torch.argmax(logprob)][0]

    def entropy(self):
        sample = self._dist.sample(self._samples)
        logprob = self.log_prob(sample)
        return -torch.mean(logprob, 0)


class OneHotDist(torchd.one_hot_categorical.OneHotCategorical):

    def __init__(self, logits=None, probs=None):
        super().__init__(logits=logits, probs=probs)

    def mode(self):
        _mode = F.one_hot(torch.argmax(super().logits, axis=-1), super().logits.shape[-1])
        return _mode.detach() + super().logits - super().logits.detach()

    def sample(self, sample_shape=(), seed=None):
        if seed is not None:
            raise ValueError('need to check')
        sample = super().sample(sample_shape)
        probs = super().probs
        while len(probs.shape) < len(sample.shape):
            probs = probs[None]
        sample += probs - probs.detach()
        return sample


class ContDist:

    def __init__(self, dist=None):
        super().__init__()
        self._dist = dist
        self.mean = dist.mean

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def entropy(self):
        return self._dist.entropy()

    def mode(self):
        return self._dist.mean

    def sample(self, sample_shape=()):
        return self._dist.rsample(sample_shape)

    def log_prob(self, x):
        return self._dist.log_prob(x)


class Bernoulli:

    def __init__(self, dist=None):
        super().__init__()
        self._dist = dist
        self.mean = dist.mean

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def entropy(self):
        return self._dist.entropy()

    def mode(self):
        _mode = torch.round(self._dist.mean)
        return _mode.detach() + self._dist.mean - self._dist.mean.detach()

    def sample(self, sample_shape=()):
        return self._dist.rsample(sample_shape)

    def log_prob(self, x):
        _logits = self._dist.base_dist.logits
        log_probs0 = -F.softplus(_logits)
        log_probs1 = -F.softplus(-_logits)

        return log_probs0 * (1 - x) + log_probs1 * x


class UnnormalizedHuber(torchd.normal.Normal):

    def __init__(self, loc, scale, threshold=1, **kwargs):
        super().__init__(loc, scale, **kwargs)
        self._threshold = threshold

    def log_prob(self, event):
        return -(torch.sqrt((event - self.mean)**2 + self._threshold**2) - self._threshold)

    def mode(self):
        return self.mean


class SafeTruncatedNormal(torchd.normal.Normal):

    def __init__(self, loc, scale, low, high, clip=1e-6, mult=1):
        super().__init__(loc, scale)
        self._low = low
        self._high = high
        self._clip = clip
        self._mult = mult

    def sample(self, sample_shape):
        event = super().sample(sample_shape)
        if self._clip:
            clipped = torch.clip(event, self._low + self._clip, self._high - self._clip)
            event = event - event.detach() + clipped.detach()
        if self._mult:
            event *= self._mult
        return event
