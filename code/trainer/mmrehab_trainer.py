import math
import pickle
import torch
import torch.nn as nn

from trainer.base_trainer import BaseTrainer
from util.logger import log
from util.utils import rotation6d_2_rot_mat, rodrigues_2_rot_mat
from util.mesh_loss import MeshLoss
from util.geodesic_loss import GeodesicLoss
from util.common_utils import compute_eucl_dist, compute_angular_dist
from util.similarity_loss import SimilarityLoss
from util.joints_similarity_loss import JointsSimilarityLoss
from util.ssim_loss import SSIMLoss
from util.s3im_loss import S3IMLoss

def gen_loss_fn(cfg):
    if cfg['fn'] == 'mse':
        return nn.MSELoss()
    if cfg['fn'] == 'mae':
        return nn.L1Loss()
    if cfg['fn'] == 'mesh':
        smplx = cfg['smplx'] if 'smplx' in cfg else True
        return_joints = cfg['return_joints'] if 'return_joints' in cfg else False
        return MeshLoss(smplx=smplx, return_joints=return_joints)
    if cfg['fn'] == 'geodesic':
        return GeodesicLoss()
    if cfg['fn'] == 'cos_sim':
        return SimilarityLoss()
    if cfg['fn'] == 'joints_sim':
        return JointsSimilarityLoss(cfg['adaptive_weight'])
    if cfg['fn'] == 'huber':
        return nn.HuberLoss()
    if cfg['fn'] == 'ce':
        return nn.CrossEntropyLoss()
    if cfg['fn'] == 'ssim':
        ks = cfg['kernel_size'] if 'kernel_size' in cfg else 11
        return SSIMLoss(n_channels=1, window_size=ks)
    if cfg['fn'] == 's3im':
        return S3IMLoss(kernel_size=11, stride=1, patch_height=16, patch_width=16)

def gen_loss_fns(losses):
    loss_fns = {}
    for loss in losses:
        loss_fns[loss['name']] = {
            'fn': gen_loss_fn(loss),
            'weight': loss['weight']
        }
    return loss_fns

def mean(list_tensors):
    return torch.mean(torch.stack(list(list_tensors)))

class MmrehabTrainer(BaseTrainer):
    def __init__(self, config, model):
        loss_fns = gen_loss_fns(config.training.losses)
        super().__init__(config, model, loss_fns)

    def calculate_total_loss(self, outputs, targets, pose_labels=None, return_joints=False, return_rot_mat=False):
        training_config = self.training_config
        smplx = training_config.smplx if 'smplx' in training_config else True

        loss_total = 0
        losses = {}
        num_joints = 24

        output, target = outputs['feats'], targets['feats']
        if len(output.shape) < 3:
            output = output.unsqueeze(1)
            target = target.unsqueeze(1)
        time_len = target.shape[1]
        output = output[:, -time_len:, :]
        for loss_name in ['kd_mse', 'kd_sim']:
            if loss_name not in self.loss_fns:
                continue
            fn = self.loss_fns[loss_name]['fn']
            w = self.loss_fns[loss_name]['weight']
            if loss_name == 'kd_mse':
                mse_loss = fn(output, target)
                losses[loss_name] = mse_loss.item()
                loss_total += w * mse_loss
            elif loss_name == 'kd_sim':
                sim_loss = fn(output.reshape((-1, output.shape[2])), target.reshape((-1, target.shape[2])))
                losses[loss_name] = sim_loss.item()
                loss_total += w * sim_loss

        output, target = outputs['smpl'], targets['smpl']
        if len(output.shape) < 3:
            output = output.unsqueeze(1)
            target = target.unsqueeze(1)
        time_len = target.shape[1]
        output = output[:, -time_len:, :]
        for loss_name in ['pose_cls', 'trans', 'pose', 'smpl', 'shape', 'joints', 'vertices', 'rotmat', 'keypoints', 'joints_similarity', 'joints_similarity2', 'adv', 'diff_joints']:
            if loss_name not in self.loss_fns:
                continue
            fn = self.loss_fns[loss_name]['fn']
            w = self.loss_fns[loss_name]['weight']
            if loss_name == 'pose_cls':
                num_poses = self.config.dataset.num_poses
                pose_cls_loss = fn(output[:, -1, :num_poses], pose_labels)
                losses[loss_name] = pose_cls_loss.item()
                loss_total += w * pose_cls_loss
                output = output[:, :, num_poses:]
            elif loss_name == 'trans':
                trans_loss = fn(output[:, :, 0:3], target[:, :, 0:3])
                losses[loss_name] = trans_loss.item()
                loss_total += w * trans_loss
            elif loss_name == 'pose':
                if smplx:
                    output_mat = [rotation6d_2_rot_mat(output[:, t, 3:-16]) for t in range(time_len)]
                    target_mat = [rodrigues_2_rot_mat(target[:, t, 3:-16]) for t in range(time_len)]
                else:
                    output_mat = [rotation6d_2_rot_mat(output[:, t, :-10]) for t in range(time_len)]
                    target_mat = target[:, :, :-10].permute((1, 0, 2))
                    if target_mat.shape[2] == 72:
                        target_mat = [rodrigues_2_rot_mat(target_mat[t]) for t in range(time_len)]
                    elif target_mat.shape[2] == 75:
                        target_mat = [rodrigues_2_rot_mat(target_mat[t][:, 3:]) for t in range(time_len)]

                pose_loss = mean([fn(output_mat[t], target_mat[t]) for t in range(time_len)])
                losses[loss_name] = pose_loss.item()
                loss_total += w * pose_loss
            elif loss_name == 'smpl':
                if smplx:
                    num_joints = 22
                    ignore_trans = training_config.ignore_trans if 'ignore_trans' in training_config else False
                    out = [fn(torch.cat((torch.zeros((output.shape[0], 3)).to(output) if ignore_trans else output[:, t, :3], output_mat[t], output[:, t, -16:]), -1),
                                        torch.cat((torch.zeros((target.shape[0], 3)).to(target) if ignore_trans else target[:, t, :3], target_mat[t], target[:, t, -16:]), -1), training_config.use_gender) for t in range(time_len)]
                else:
                    out = [fn(torch.cat((output_mat[t], output[:, t, -10:]), -1),
                                        torch.cat((target_mat[t], target[:, t, -10:]), -1), training_config.use_gender) for t in range(time_len)]

                v_loss = mean([v[0] for v in out])
                j_loss = mean([v[1] for v in out])
                if len(out[0]) > 2:
                    pred_joints = [v[2] for v in out]
                    target_joints = [v[3] for v in out]
            elif loss_name == 'shape':
                if smplx:
                    shape_loss = mean([fn(output[:, t, -16:], target[:, t, -16:]) for t in range(time_len)])
                    loss_total += w * shape_loss
                else:
                    shape_loss = mean([fn(output[:, t, -10:], target[:, t, -10:]) for t in range(time_len)])
                    loss_total += w * shape_loss
                losses[loss_name] = shape_loss.item()
            elif loss_name == 'joints':
                losses[loss_name] = j_loss.item()
                loss_total += w * j_loss
            elif loss_name == 'vertices':
                losses[loss_name] = v_loss.item()
                loss_total += w * v_loss
            elif loss_name == 'rotmat':
                rotmat_loss = mean([fn(output_mat[t], target_mat[t]) for t in range(time_len)])
                losses[loss_name] = rotmat_loss.item()
                loss_total += w * rotmat_loss
            elif loss_name == 'keypoints':
                kp_loss = fn(output, target)
                losses[loss_name] = kp_loss.item()
                loss_total += w * kp_loss
            elif loss_name == 'joints_similarity':
                if 'stacked_pred_joints' not in locals():
                    stacked_pred_joints = torch.stack(pred_joints, dim=1)
                    stacked_target_joints = torch.stack(target_joints, dim=1)
                js_loss = mean([fn(stacked_pred_joints[:, :, k, a], stacked_target_joints[:, :, k, a]) for k in range(stacked_pred_joints.shape[2]) for a in range(stacked_pred_joints.shape[3])])
                losses[loss_name] = js_loss.item()
                loss_total += w * js_loss
            elif loss_name == 'joints_similarity2':
                if 'stacked_pred_joints' not in locals():
                    stacked_pred_joints = torch.stack(pred_joints, dim=1)
                    stacked_target_joints = torch.stack(target_joints, dim=1)
                js_loss = fn(stacked_pred_joints, stacked_target_joints)
                losses[loss_name] = js_loss.item()
                loss_total += w * js_loss
            elif loss_name == 'diff_joints':
                if 'stacked_pred_joints' not in locals():
                    stacked_pred_joints = torch.stack(pred_joints, dim=1)
                    stacked_target_joints = torch.stack(target_joints, dim=1)
                diff_pred_joints = stacked_pred_joints.diff(dim=1)
                diff_target_joints = stacked_target_joints.diff(dim=1)
                dj_loss = mean([fn(diff_pred_joints[:, :, k, a], diff_target_joints[:, :, k, a]) for k in range(diff_pred_joints.shape[2]) for a in range(diff_pred_joints.shape[3])])
                losses[loss_name] = dj_loss.item()
                loss_total += w * dj_loss
            elif loss_name == 'adv':
                if hasattr(self.model, 'forward_discriminator'):
                    pred_pose = torch.vstack(list(output_mat)).view(-1, num_joints, 3, 3)
                    pred_shape = output[:, :, -10:].reshape(-1, 10)
                    real_pose = torch.vstack(list(target_mat)).view(-1, num_joints, 3, 3)
                    real_shape = target[:, :, -10:].reshape(-1, 10)
                    cat_pose = torch.cat((pred_pose, real_pose), 0)
                    cat_shape = torch.cat((pred_shape, real_shape), 0)
                    pred = self.model.forward_discriminator(cat_pose, cat_shape)
                    B = pred.shape[0]
                    N = pred.shape[1]
                    adv_label = torch.vstack((torch.zeros((B // 2, N)), torch.ones((B // 2, N)))).to(pred)
                    adv_loss = fn(pred, adv_label)
                    losses[loss_name] = adv_loss.item()
                    loss_total += w * adv_loss
            if loss_name in losses:
                if math.isnan(losses[loss_name]):
                    log(f'NaN: {loss_name}')

        mid_res = {}
        if return_joints:
            mid_res['joints'] = (torch.stack(target_joints, dim=1), torch.stack(pred_joints, dim=1))
        if return_rot_mat:
            if 'target_mat' not in locals():
                if smplx:
                    output_mat = [rotation6d_2_rot_mat(output[:, t, 3:-16]) for t in range(time_len)]
                    target_mat = [rodrigues_2_rot_mat(target[:, t, 3:-16]) for t in range(time_len)]
                else:
                    output_mat = [rotation6d_2_rot_mat(output[:, t, :-10]) for t in range(time_len)]
                    target_mat = target[:, :, :-10].permute((1, 0, 2))
                    if target_mat.shape[2] == 72:
                        target_mat = [rodrigues_2_rot_mat(target_mat[t]) for t in range(time_len)]
                    elif target_mat.shape[2] == 75:
                        target_mat = [rodrigues_2_rot_mat(target_mat[t][:, 3:]) for t in range(time_len)]
            mid_res['rotmat'] = (torch.vstack(list(target_mat)).view(-1, num_joints, 3, 3), torch.vstack(list(output_mat)).view(-1, num_joints, 3, 3))
        return loss_total, losses, mid_res

    def train_step(self, samples):
        input, target, pos = samples
        pose_labels = None
        if len(pos) >= 3:
            pose_labels = pos[2].to(self.device)

        if isinstance(input, dict):
            for k, v in input.items():
                input[k] = v.to(self.device)
            clip = next(iter(input.values()))
        else:
            input = input.to(self.device)
            clip = input
        for k, v in target.items():
            target[k] = v.to(self.device)

        b_size = clip.shape[0]

        output_smpl, feats, depth_feats = self.model(input, { 'depth_mesh': target.get('depth_mesh', None) })
        outputs = {
            'smpl': output_smpl,
            'feats': feats
        }
        target['feats'] = depth_feats

        loss_total, losses, _ = self.calculate_total_loss(outputs, target, pose_labels=pose_labels)
        
        return loss_total, losses, b_size

    def eval_step(self, samples, save_path=None):
        input, target, pos = samples
        pose_labels = None
        if len(pos) >= 3:
            pose_labels = pos[2].to(self.device)

        if isinstance(input, dict):
            for k, v in input.items():
                input[k] = v.to(self.device)
            clip = next(iter(input.values()))
        else:
            input = input.to(self.device)
            clip = input
        for k, v in target.items():
            target[k] = v.to(self.device)

        b_size = clip.shape[0]

        output_smpl, feats, depth_feats = self.model(input, { 'depth_mesh': target.get('depth_mesh', None) })
        outputs = {
            'smpl': output_smpl,
            'feats': feats
        }
        target['feats'] = depth_feats

        loss_total, losses, mid_res = self.calculate_total_loss(outputs, target, return_joints=True, return_rot_mat=True, pose_labels=pose_labels)
        
        joints = mid_res.get('joints', None)
        rotmat = mid_res.get('rotmat', None)
        
        if joints is not None and rotmat is not None:
            n_joints = joints[0].shape[-2]
            gt_joints = joints[0].view(-1, n_joints, 3).detach().cpu().numpy()
            es_joints = joints[1].view(-1, n_joints, 3).detach().cpu().numpy()
            mpjpe = compute_eucl_dist(gt_joints, es_joints).mean()
            pa_mpjpe = compute_eucl_dist(gt_joints, es_joints, procrustes=True).mean()
            
            n_joints_rot = rotmat[0].shape[-3]
            angle_err = compute_angular_dist(rotmat[0].view(-1, n_joints_rot*3*3).detach().cpu().numpy(), rotmat[1].view(-1, n_joints_rot*3*3).detach().cpu().numpy(), rep='rotmat').mean()
            
            losses['pa_mpjpe'] = pa_mpjpe
            losses['mpjpe'] = mpjpe
            losses['angle'] = angle_err
            
        return loss_total.item(), losses, b_size
