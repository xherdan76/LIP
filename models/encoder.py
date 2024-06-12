import numpy as np
import open3d.ml.torch as ml3d
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.utils as utils
from torch import distributions as torchd


class ParticleEncoder(torch.nn.Module):

    def __init__(
            self,
            kernel_size=[4, 4, 4],
            radius_scale=1.5,
            coordinate_mapping='ball_to_cube_volume_preserving',
            interpolation='linear',
            use_window=True,
            particle_radius=0.025,
            timestep=1 / 50,
            other_feats_channels=0,
    ):
        super().__init__()
        self.layer_channels = [32, 64, 64, 3]
        self.kernel_size = kernel_size
        self.radius_scale = radius_scale
        self.coordinate_mapping = coordinate_mapping
        self.interpolation = interpolation
        self.use_window = use_window
        self.particle_radius = particle_radius
        self.filter_extent = np.float32(self.radius_scale * 6 *
                                        self.particle_radius)
        self.timestep = timestep

        self._all_convs = []

        def window_poly6(r_sqr):
            return torch.clamp((1 - r_sqr)**3, 0, 1)

        def Conv(name, activation=None, **kwargs):
            conv_fn = ml3d.layers.ContinuousConv

            window_fn = None
            if self.use_window == True:
                window_fn = window_poly6

            conv = conv_fn(kernel_size=self.kernel_size,
                           activation=activation,
                           align_corners=True,
                           interpolation=self.interpolation,
                           coordinate_mapping=self.coordinate_mapping,
                           normalize=False,
                           window_function=window_fn,
                           radius_search_ignore_query_points=True,
                           **kwargs)

            self._all_convs.append((name, conv))
            return conv

        self.conv0_fluid = Conv(name="conv0_fluid",
                                in_channels=4 + other_feats_channels,
                                filters=self.layer_channels[0],
                                activation=None)
        self.conv0_obstacle = Conv(name="conv0_obstacle",
                                   in_channels=3,
                                   filters=self.layer_channels[0],
                                   activation=None)
        self.dense0_fluid = torch.nn.Linear(in_features=4 +
                                            other_feats_channels,
                                            out_features=self.layer_channels[0])
        torch.nn.init.xavier_uniform_(self.dense0_fluid.weight)
        torch.nn.init.zeros_(self.dense0_fluid.bias)

        self.convs = []
        self.denses = []
        for i in range(1, len(self.layer_channels)):
            in_ch = self.layer_channels[i - 1]
            if i == 1:
                in_ch *= 3
            out_ch = self.layer_channels[i]
            dense = torch.nn.Linear(in_features=in_ch, out_features=out_ch)
            torch.nn.init.xavier_uniform_(dense.weight)
            torch.nn.init.zeros_(dense.bias)
            setattr(self, 'dense{0}'.format(i), dense)
            conv = Conv(name='conv{0}'.format(i),
                        in_channels=in_ch,
                        filters=out_ch,
                        activation=None)
            setattr(self, 'conv{0}'.format(i), conv)
            self.denses.append(dense)
            self.convs.append(conv)


    def compute_correction(self,
                           pos,
                           vel,
                           other_feats,
                           box,
                           box_feats,
                           fixed_radius_search_hash_table=None):
        """Expects that the pos and vel has already been updated with gravity and velocity"""

        # compute the extent of the filters (the diameter)
        filter_extent = torch.tensor(self.filter_extent)

        fluid_feats = [torch.ones_like(pos[:, 0:1]), vel]
        if not other_feats is None:
            fluid_feats.append(other_feats)
        fluid_feats = torch.cat(fluid_feats, axis=-1)

        self.ans_conv0_fluid = self.conv0_fluid(fluid_feats, pos, pos,
                                                filter_extent)
        self.ans_dense0_fluid = self.dense0_fluid(fluid_feats)
        self.ans_conv0_obstacle = self.conv0_obstacle(box_feats, box, pos,
                                                      filter_extent)

        feats = torch.cat([
            self.ans_conv0_obstacle, self.ans_conv0_fluid, self.ans_dense0_fluid
        ],
                          axis=-1)

        self.ans_convs = [feats]
        for conv, dense in zip(self.convs, self.denses):
            inp_feats = F.relu(self.ans_convs[-1])
            ans_conv = conv(inp_feats, pos, pos, filter_extent)
            ans_dense = dense(inp_feats)
            if ans_dense.shape[-1] == self.ans_convs[-1].shape[-1]:
                ans = ans_conv + ans_dense + self.ans_convs[-1]
            else:
                ans = ans_conv + ans_dense
            self.ans_convs.append(ans)

        # compute the number of fluid neighbors.
        # this info is used in the loss function during training.
        self.num_fluid_neighbors = ml3d.ops.reduce_subarrays_sum(
            torch.ones_like(self.conv0_fluid.nns.neighbors_index,
                            dtype=torch.float32),
            self.conv0_fluid.nns.neighbors_row_splits)

        self.last_features = self.ans_convs[-2]

        # scale to better match the scale of the output distribution
        self.pos_correction = (1.0 / 128) * self.ans_convs[-1]
        return self.pos_correction, self.last_features

    def forward(self, inputs, fixed_radius_search_hash_table=None):
        """computes 1 simulation timestep
        inputs: list or tuple with (pos,vel,feats,box,box_feats)
          pos and vel are the positions and velocities of the fluid particles.
          feats is reserved for passing additional features, use None here.
          box are the positions of the static particles and box_feats are the
          normals of the static particles.
        """
        pos, vel, feats, box, box_feats = inputs
        pos_correction, particle_feat = self.compute_correction(
            pos, vel, feats, box, box_feats, fixed_radius_search_hash_table)

        return particle_feat


class GaussianGRU(nn.Module):
    def __init__(self, input_size=64, output_size=8, hidden_size=64, n_layers=1, mean_act='none'):
        super().__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        assert input_size == hidden_size
        # self.embed = nn.Linear(input_size, hidden_size)
        self.gru = nn.ModuleList([nn.GRUCell(hidden_size, hidden_size) for i in range(self.n_layers)])
        self.mu_net = nn.Linear(hidden_size, output_size)
        self.logvar_net = nn.Linear(hidden_size, output_size)
        self.hidden = self.init_hidden(0)

        self._mean_act = mean_act
        self._std_act = 'sigmoid2'
        self._min_std = 0.02

    def init_hidden(self, batch_size):
        hidden = []
        for i in range(self.n_layers):
            hidden.append(torch.zeros(batch_size, self.hidden_size).cuda())
        self.hidden = hidden
        return hidden

    def forward(self, input):
        # embedded = self.embed(input)
        # h_in = embedded
        h_in = input
        for i in range(self.n_layers):
            self.hidden[i] = self.gru[i](h_in, self.hidden[i])
            h_in = self.hidden[i]
        mean, std = self._suff_stats_layer(h_in)
        dist = utils.ContDist(torchd.independent.Independent(torchd.normal.Normal(mean, std), 1))
        z = dist.sample()
        stat = {'mean': mean, 'std': std}
        return z, stat #dist._dist

    def _suff_stats_layer(self, x):
        mean = self.mu_net(x)
        std = self.logvar_net(x)
        mean = {
            'none': lambda: mean,
            'tanh5': lambda: 5.0 * torch.tanh(mean / 5.0),
            'tanh': lambda: 1.0 * torch.tanh(mean),
        }[self._mean_act]()
        std = {
            'softplus': lambda: torch.softplus(std),
            'abs': lambda: torch.abs(std + 1),
            'sigmoid': lambda: torch.sigmoid(std),
            'sigmoid2': lambda: 2 * torch.sigmoid(std / 2),
        }[self._std_act]()
        std = std + self._min_std
        return mean, std

    def get_dist(self, state, dtype=None):
        mean, std = state['mean'], state['std']
        dist = utils.ContDist(torchd.independent.Independent(torchd.normal.Normal(mean, std), 1))
        return dist

    def stop_gradient(self):
        for i in range(len(self.hidden)):
            self.hidden[i] = self.hidden[i].detach()