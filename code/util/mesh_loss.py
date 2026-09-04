import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.nn.modules.loss import _Loss

from util.utils import rodrigues_2_rot_mat
from smpl.smpl_utils_extend import SMPL as SMPLModel

try:
    from human_body_prior.body_model.body_model import BodyModel
    from human_body_prior.body_model.lbs import lbs
except ImportError:
    BodyModel = None
    lbs = None


SMPLX_MODEL_NEUTRAL_PATH = './smpl/smplx/SMPLX_NEUTRAL.npz'
SMPLX_MODEL_FEMALE_PATH = './smpl/smplx/SMPLX_FEMALE.npz'
SMPLX_MODEL_MALE_PATH = './smpl/smplx/SMPLX_MALE.npz'

SMPL_MODEL_FEMALE_PATH = './smpl/smpl_f.pkl'
SMPL_MODEL_MALE_PATH = './smpl/smpl_m.pkl'


class MeshLoss(_Loss):
    smplx_model = [None, None, None]
    smpl_model = [None, None]
    def __init__(self, device: torch.device = torch.device('cpu'), smplx=True, size_average=None, reduce=None, reduction: str = 'mean', scale: float = 1, return_joints=False) -> None:
        super().__init__(size_average=size_average, reduce=reduce, reduction=reduction)
        self.smplx = smplx
        if smplx:
            if self.smplx_model[0] is None:
                self.smplx_model[0] = SMPLXModel(bm_fname=SMPLX_MODEL_FEMALE_PATH, num_betas=16, num_expressions=0)
            if self.smplx_model[1] is None:
                self.smplx_model[1] = SMPLXModel(bm_fname=SMPLX_MODEL_MALE_PATH, num_betas=16, num_expressions=0)
            if self.smplx_model[2] is None:
                self.smplx_model[2] = SMPLXModel(bm_fname=SMPLX_MODEL_NEUTRAL_PATH, num_betas=16, num_expressions=0)
        else:
            if self.smpl_model[0] is None:
                self.smpl_model[0] = SMPLModel(SMPL_MODEL_FEMALE_PATH)
            if self.smpl_model[1] is None:
                self.smpl_model[1] = SMPLModel(SMPL_MODEL_MALE_PATH)
        self.scale = scale
        self.return_joints = return_joints

    def forward(self, input: torch.Tensor, target: torch.Tensor, use_gender: int = 0, train: bool = True, use_rodrigues=False) -> torch.Tensor:
        if self.smplx:
            return self.forward_smplx(input, target, use_gender, train, use_rodrigues)
        return self.forward_smpl(input, target, use_gender)

    def forward_smplx(self, input: torch.Tensor, target: torch.Tensor, use_gender: int = 0, train: bool = True, use_rodrigues=False):
        _input = input * self.scale
        _target = target * self.scale

        if not use_gender:
            input_model = target_model = self.smplx_model[2]
        else:
            input_model = target_model = self.smplx_model[0 if target[0][-1] < 0.5 else 1]

        input_result = input_model(pose_body=_input[:,:-16], betas=_input[:,-16:], use_rodrigues=use_rodrigues)
        input_verts = input_result['vertices']
        input_joints = input_result['joints']

        target_result = target_model(pose_body=_target[:,:-16], betas=_target[:,-16:], use_rodrigues=use_rodrigues)
        target_verts = target_result['vertices']
        target_joints = target_result['joints']

        per_joint_err = torch.norm((input_joints - target_joints), dim=-1)
        per_vertex_err = torch.norm((input_verts - target_verts), dim=-1)

        if train:
            v_loss = F.l1_loss(input_verts, target_verts, reduction=self.reduction)
            j_loss = F.l1_loss(input_joints, target_joints, reduction=self.reduction)
        else:
            v_loss = torch.sqrt(F.mse_loss(input_verts, target_verts, reduction=self.reduction))
            j_loss = torch.sqrt(F.mse_loss(input_joints, target_joints, reduction=self.reduction))

        if self.return_joints:
            return (v_loss, j_loss, input_joints, target_joints)
        return (v_loss, j_loss)

    def forward_smpl(self, input: torch.Tensor, target: torch.Tensor, use_gender: int = 0):
        _input = input * self.scale
        _target = target * self.scale

        if not use_gender:
            input_model = target_model = self.smpl_model[1]
        else:
            input_model = target_model =  self.smpl_model[0 if target[0][-1] < 0.5 else 1]

        input_verts, input_joints = input_model(pose=_input[:,:-10].view(-1, 24, 3, 3), betas=_input[:,-10:], trans=torch.zeros((input.shape[0], 3)).to(input))
        target_verts, target_joints = target_model(pose=_target[:,:-10].view(-1, 24, 3, 3), betas=_target[:,-10:], trans=torch.zeros((input.shape[0], 3)).to(input))

        v_loss = F.l1_loss(input_verts, target_verts, reduction=self.reduction)
        j_loss = F.l1_loss(input_joints, target_joints, reduction=self.reduction)
        if self.return_joints:
            return (v_loss, j_loss, input_joints, target_joints)
        return (v_loss, j_loss)


class SMPLXModel(BodyModel if BodyModel is not None else nn.Module):
    def __init__(self, bm_fname=SMPLX_MODEL_NEUTRAL_PATH, num_betas=16, num_expressions=0, **kwargs):
        if BodyModel is None:
            raise ImportError(
                'SMPL-X support requires the separately licensed human_body_prior package.'
            )
        super().__init__(bm_fname=bm_fname, num_betas=num_betas, num_expressions=num_expressions, **kwargs)

    def forward(self, pose_body, betas, use_rodrigues=True):
        device = pose_body.device
        for name in ['init_pose_hand', 'init_pose_jaw','init_pose_eye', 'init_v_template', 'init_expression',
                    'shapedirs', 'exprdirs', 'posedirs', 'J_regressor', 'kintree_table', 'weights', ]:
            _tensor = getattr(self, name)
            setattr(self, name, _tensor.to(device))

        batch_size = pose_body.shape[0]
        trans = pose_body[:, :3]
        pose_hand = self.init_pose_hand.expand(batch_size, -1)
        pose_jaw = self.init_pose_jaw.expand(batch_size, -1)
        pose_eye = self.init_pose_eye.expand(batch_size, -1)
        v_template = self.init_v_template.expand(batch_size, -1, -1)
        expression = self.init_expression.expand(batch_size, -1)

        init_pose = torch.cat([pose_jaw, pose_eye, pose_hand], dim=-1)
        if not use_rodrigues:
            init_pose = rodrigues_2_rot_mat(init_pose)
        full_pose = torch.cat([pose_body[:, 3:], init_pose], dim=-1)
        shape_components = torch.cat([betas, expression], dim=-1)
        shapedirs = torch.cat([self.shapedirs, self.exprdirs], dim=-1)

        verts, joints = lbs(betas=shape_components, pose=full_pose, v_template=v_template,
                        shapedirs=shapedirs, posedirs=self.posedirs, J_regressor=self.J_regressor,
                        parents=self.kintree_table[0].long(), lbs_weights=self.weights, pose2rot=use_rodrigues)

        joints = joints + trans.unsqueeze(dim=1)
        verts = verts + trans.unsqueeze(dim=1)
        return dict(vertices=verts, joints=joints)

