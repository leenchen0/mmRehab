from functools import reduce

import numpy as np
import torch.nn as nn
import torch
from torch.autograd import Function
import torch.nn.functional as F

from util.resnet import ResNetLayer, conv3x3, conv7x7


class MmRehabKD(nn.Module):
    def __init__(self, config, in_channels, out_channels):
        super().__init__()

        model_config = config.model[config.model.backbone]
        encoder_out = model_config.encoder_out
        n_filters = model_config.n_filters
        embed_dim = model_config.embed_dim
        num_heads = model_config.num_heads
        dropout_rate = model_config.dropout if 'dropout' in model_config else 0
        dataset_config = config.dataset[config.dataset.format] if 'dataset' in config else config.datasets[0][config.datasets[0].format]
        heatmaps = dataset_config.model_input_data if 'model_input_data' in dataset_config else dataset_config.input_data
        num_heatmaps = len(heatmaps)
        test_input_data = dataset_config.input_data

        self.encoders = {}
        self.hm_remapping = {}
        for i, hm in enumerate(heatmaps):
            self.encoders[hm] = Encoder(in_channels, encoder_out, n_filters=n_filters, embed_dim=embed_dim, num_heads=num_heads)
            if not isinstance(test_input_data[i], (list, tuple)):
                test_input_data[i] = [test_input_data[i]]
            for field in test_input_data[i]:
                self.hm_remapping[field] = hm
        self.encoders = nn.ModuleDict(self.encoders)

        dim_embed = encoder_out * num_heatmaps
        lstm_dim = model_config.lstm_dim
        num_layers_rnn = 1 if 'num_layers_lstm' not in model_config else model_config.num_layers_lstm
        self.rnn = nn.LSTM(encoder_out * num_heatmaps, lstm_dim // 2, num_layers=num_layers_rnn, bidirectional=True, batch_first=True)

        dim_embed_teacher = config.model.teacher.model.encoder_out
        self.projection = nn.Sequential(
            nn.LayerNorm(dim_embed),
            nn.Linear(dim_embed, dim_embed_teacher),
        )
        share_head = model_config.share_head if 'share_head' in model_config else False
        self.teacher_model = Teacher(config, in_channels, out_channels)
        self.teacher_model.load_state_dict(torch.load(config.model.teacher.path, 'cpu'))
        # Freeze teacher model
        for p in self.teacher_model.parameters():
            p.requires_grad = False

        if share_head:
            self.mesh_head = self.teacher_model.estimator.projection
            for p in self.mesh_head.parameters():
                p.requires_grad = False
        else:
            self.mesh_head = MeshHead(model_config, dim_embed_teacher, out_channels)

        if 'discriminator' in config.model and config.model.discriminator:
            self.discriminator = Discriminator(model_config.num_joints)

    def forward(self, x, depth=None):
        out_maps = [torch.stack([self.encoders[self.hm_remapping[k]](v[:, t:(t+1), :, :]) for t in range(v.shape[1])], dim=1) for k, v in x.items()]
        out_encoders = torch.cat(out_maps, dim=-1)
        # out, _ = self.rnn(out_encoders)
        feats, _ = self.rnn(out_encoders)
        feats = self.projection(feats)
        mesh = self.mesh_head(feats)

        if depth is None:
            return mesh

        t_mesh, t_feats = self.teacher_model(depth)
        return mesh, feats, t_feats

    def forward_discriminator(self, pose, shape):
        res = self.discriminator(pose, shape)
        return res


class MeshHead(nn.Module):
    def __init__(self, model_config, dim_embed, dim_out) -> None:
        super(MeshHead, self).__init__()
        self.projection = nn.Sequential(
            nn.LayerNorm(dim_embed),
            nn.Linear(dim_embed, dim_out * 2),
            nn.GELU(),

            nn.Linear(dim_out * 2, dim_out)
        )

    def forward(self, features):
        res = self.projection(features)

        return res


class MmRehab(nn.Module):
    def __init__(self, config, in_channels, out_channels):
        super().__init__()

        model_config = config.model.teacher.model
        encoder_out = model_config.encoder_out
        n_filters = model_config.n_filters
        embed_dim = model_config.embed_dim
        num_heads = model_config.num_heads
        heatmaps = config.model.teacher.model_input_data
        num_heatmaps = len(heatmaps)
        test_input_data = config.model.teacher.input_data

        self.encoders = {}
        self.hm_remapping = {}
        for i, hm in enumerate(heatmaps):
            self.encoders[hm] = Encoder(in_channels, encoder_out, n_filters=n_filters, embed_dim=embed_dim, num_heads=num_heads)
            if not isinstance(test_input_data[i], (list, tuple)):
                test_input_data[i] = [test_input_data[i]]
            for field in test_input_data[i]:
                self.hm_remapping[field] = hm
        self.encoders = nn.ModuleDict(self.encoders)

        lstm_dim = model_config.lstm_dim
        num_layers_rnn = 1 if 'num_layers_lstm' not in model_config else model_config.num_layers_lstm
        self.rnn = nn.LSTM(encoder_out * num_heatmaps, lstm_dim // 2, num_layers=num_layers_rnn, bidirectional=True, batch_first=True)
        self.projection = nn.Sequential(
            nn.LayerNorm(encoder_out * num_heatmaps),
            nn.Linear(encoder_out * num_heatmaps, out_channels * 2),
            nn.GELU(),

            nn.Linear(out_channels * 2, out_channels)
        )

    def forward(self, x):
        out_maps = [torch.stack([self.encoders[self.hm_remapping[k]](v[:, t:(t+1), :, :]) for t in range(v.shape[1])], dim=1) for k, v in x.items()]
        out_encoders = torch.cat(out_maps, dim=-1)
        # out, _ = self.rnn(out_encoders)
        out, _ = self.rnn(out_encoders)
        res = self.projection(out)

        return res, out


class Teacher(nn.Module):
    def __init__(self, config, in_channels, out_channels):
        super(Teacher, self).__init__()

        self.estimator = MmRehab(config, in_channels, out_channels)

    def forward(self, x):
        return self.estimator(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, n_filters, embed_dim, num_heads) -> None:
        super(Encoder, self).__init__()

        self.cnn= nn.Sequential(
            ResNetLayer(in_channels, n_filters[0], conv=conv7x7),
            *[ResNetLayer(n_filters[i - 1], n_filters[i], conv=conv7x7) for i in range(1, len(n_filters))]
        )
        self.linear = nn.Sequential(
            nn.Linear(n_filters[-1], embed_dim),
            nn.ReLU(),

            nn.Linear(embed_dim, out_channels)
        )

    def forward(self, x):
        out_cnn = self.cnn(x)
        flatten_cnn = torch.flatten(out_cnn, 2).sum(dim=2)
        out = self.linear(flatten_cnn)
        return out


class Discriminator(nn.Module):
    def __init__(self, num_joints):
        super(Discriminator, self).__init__()

        self.num_joints = num_joints
        # poses_alone
        self.D_conv1 = nn.Conv2d(9, 32, kernel_size=1)
        nn.init.xavier_uniform_(self.D_conv1.weight)
        nn.init.zeros_(self.D_conv1.bias)
        self.relu = nn.ReLU(inplace=True)
        self.D_conv2 = nn.Conv2d(32, 32, kernel_size=1)
        nn.init.xavier_uniform_(self.D_conv2.weight)
        nn.init.zeros_(self.D_conv2.bias)
        pose_out = []
        for i in range(self.num_joints):
            pose_out_temp = nn.Linear(32, 1)
            nn.init.xavier_uniform_(pose_out_temp.weight)
            nn.init.zeros_(pose_out_temp.bias)
            pose_out.append(pose_out_temp)
        self.pose_out = nn.ModuleList(pose_out)

        # betas
        self.betas_fc1 = nn.Linear(10, 10)
        nn.init.xavier_uniform_(self.betas_fc1.weight)
        nn.init.zeros_(self.betas_fc1.bias)
        self.betas_fc2 = nn.Linear(10, 5)
        nn.init.xavier_uniform_(self.betas_fc2.weight)
        nn.init.zeros_(self.betas_fc2.bias)
        self.betas_out = nn.Linear(5, 1)
        nn.init.xavier_uniform_(self.betas_out.weight)
        nn.init.zeros_(self.betas_out.bias)

        # poses_joint
        self.D_alljoints_fc1 = nn.Linear(32*self.num_joints, 1024)
        nn.init.xavier_uniform_(self.D_alljoints_fc1.weight)
        nn.init.zeros_(self.D_alljoints_fc1.bias)
        self.D_alljoints_fc2 = nn.Linear(1024, 1024)
        nn.init.xavier_uniform_(self.D_alljoints_fc2.weight)
        nn.init.zeros_(self.D_alljoints_fc2.bias)
        self.D_alljoints_out = nn.Linear(1024, 1)
        nn.init.xavier_uniform_(self.D_alljoints_out.weight)
        nn.init.zeros_(self.D_alljoints_out.bias)

    def forward(self, poses, betas, alpha=1.0):
        """
        Forward pass of the discriminator.
        Args:
            poses (torch.Tensor): Tensor of shape (B, 24, 3, 3) containing a batch of SMPL body poses (excluding the global orientation).
            betas (torch.Tensor): Tensor of shape (B, 10) containign a batch of SMPL beta coefficients.
        Returns:
            torch.Tensor: Discriminator output with shape (B, 25)
        """
        poses = ReverseLayerF.apply(poses, alpha)
        betas = ReverseLayerF.apply(betas, alpha)
        #import ipdb; ipdb.set_trace()
        #bn = poses.shape[0]
        # poses B x 207
        #poses = poses.reshape(bn, -1)
        # poses B x num_joints x 1 x 9
        poses = poses.reshape(-1, self.num_joints, 1, 9)
        bn = poses.shape[0]
        # poses B x 9 x num_joints x 1
        poses = poses.permute(0, 3, 1, 2).contiguous()

        # poses_alone
        poses = self.D_conv1(poses)
        poses = self.relu(poses)
        poses = self.D_conv2(poses)
        poses = self.relu(poses)

        poses_out = []
        for i in range(self.num_joints):
            poses_out_ = self.pose_out[i](poses[:, :, i, 0])
            poses_out.append(poses_out_)
        poses_out = torch.cat(poses_out, dim=1)

        # betas
        betas = self.betas_fc1(betas)
        betas = self.relu(betas)
        betas = self.betas_fc2(betas)
        betas = self.relu(betas)
        betas_out = self.betas_out(betas)

        # poses_joint
        poses = poses.reshape(bn,-1)
        poses_all = self.D_alljoints_fc1(poses)
        poses_all = self.relu(poses_all)
        poses_all = self.D_alljoints_fc2(poses_all)
        poses_all = self.relu(poses_all)
        poses_all_out = self.D_alljoints_out(poses_all)

        disc_out = torch.cat((poses_out, betas_out, poses_all_out), 1)
        return disc_out


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

