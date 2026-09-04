import os
import random
import pickle
import json5
import math

import quaternion
import numpy as np
import torch

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

def make_folder(filename):
    folder_path = os.path.dirname(filename)
    if not os.path.exists(folder_path):
        try:
            os.makedirs(folder_path)
        except Exception as e:
            print(f"An exception occurred: {e}")

def load_config(root_path):
    with open(os.path.join(root_path, 'config.json'), encoding="utf8") as f:
        conf = json5.load(f)
    return convert_to_arrt_dict(conf['args'])

def get_labels_from_softmax(output):
    return output.max(dim=1).indices.cpu().detach().numpy()

def convert_to_arrt_dict(d):
    for k in d.keys():
        if isinstance(d[k], (list, tuple)):
            d[k] = [convert_to_arrt_dict(i) if isinstance(i, dict) else i for i in d[k]]
        elif isinstance(d[k], dict):
            d[k] = convert_to_arrt_dict(d[k])
    return AttrDict(**d)

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def cal_user_range_by_task_no(num_users, num_tasks, task_no):
    ufrom = num_users // num_tasks * task_no + min(task_no, num_users % num_tasks)
    uto = num_users // num_tasks * (task_no + 1) + min(task_no + 1, num_users % num_tasks)
    return ufrom, uto

# def procrustes_analysis(reference, target):
#     centroid_ref = np.mean(reference, axis=0)
#     centroid_target = np.mean(target, axis=0)

#     centered_ref = reference - centroid_ref
#     centered_target = target - centroid_target

#     U, _, Vt = np.linalg.svd(np.dot(centered_target.T, centered_ref))
#     R = np.dot(U, Vt)
#     t = centroid_ref - np.dot(R, centroid_target)

#     return R, t

# def calculate_pa_mpjpe(reference, target):
#     R, t = procrustes_analysis(reference, target)
#     transformed_target = np.dot(target, R.T) + t
#     joint_errors = np.linalg.norm(transformed_target - reference, axis=-1)
#     mpjpe = np.mean(joint_errors)

#     return mpjpe

def _procrustes(X, Y, compute_optimal_scale=True):
    """
    A port of MATLAB's `procrustes` function to Numpy.
    Adapted from http://stackoverflow.com/a/18927641/1884420
    Args
      X: array NxM of targets, with N number of points and M point dimensionality
      Y: array NxM of inputs
      compute_optimal_scale: whether we compute optimal scale or force it to be 1
    Returns:
      d: squared error after transformation
      Z: transformed Y
      T: computed rotation
      b: scaling
      c: translation
    """
    muX = X.mean(0)
    muY = Y.mean(0)
    X0 = X - muX
    Y0 = Y - muY
    ssX = (X0 ** 2.).sum()
    ssY = (Y0 ** 2.).sum()
    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)
    # scale to equal (unit) norm
    X0 = X0 / normX
    Y0 = Y0 / normY
    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)
    # Make sure we have a rotation
    detT = np.linalg.det(T)
    V[:, -1] *= np.sign(detT)
    s[-1] *= np.sign(detT)
    T = np.dot(V, U.T)
    traceTA = s.sum()
    if compute_optimal_scale:  # Compute optimum scaling of Y.
        b = traceTA * normX / normY
        d = 1 - traceTA ** 2
        Z = normX * traceTA * np.dot(Y0, T) + muX
    else:  # If no scaling allowed
        b = 1
        d = 1 + ssY / ssX - 2 * traceTA * normY / normX
        Z = normY * np.dot(Y0, T) + muX
    c = muX - b * np.dot(muY, T)
    return d, Z, T, b, c

def compute_eucl_dist(kp3d, kp3d_hat, procrustes=False):
    """
    Compute and store the Euclidean distance.
    :param kp3d: A tensor of shape (N, J, 3).
    :param kp3d_hat: A tensor of shape (N, J, 3).
    :param procrustes: Whether or not to align the keypoints with Procrustes analysis before computing the error.
    """
    if procrustes:
        # Align estimate with reference.
        kps_hat = kp3d_hat
        kps_gt = kp3d
        n, j = kps_hat.shape[0], kps_hat.shape[1]
        kps_hat_p = []
        for i in range(n):
            d, Z, T, b, c = _procrustes(kps_gt[i], kps_hat[i])
            kps_hat_p.append(Z)
        kps_hat_p = np.stack(kps_hat_p)
    else:
        kps_hat_p = kp3d_hat
        kps_gt = kp3d

    # Compute Euclidean distance and store it.
    diff = kps_gt - kps_hat_p
    eucl_dist = np.sqrt(np.sum(diff * diff, axis=-1))

    return eucl_dist

# def rot_error(r_gt, r_est):
#     d = abs(math.acos((np.trace(np.dot(np.linalg.inv(r_gt), r_est)) - 1) / 2))
#     return d * 180 / math.pi

def compute_angular_dist(pose, pose_hat, rep='aa'):
    """
    Compute and store the angular error.
    :param pose: A tensor of shape (..., N_JOINTS*DOF).
    :param pose_hat: A tensor of the same shape as `pose`.
    :param rep: The representation of the input, 'aa' or 'rotmat'.
    """
    assert rep in ['aa', 'rotmat']
    dof = 3 if rep == 'aa' else 9
    n_joints = pose.shape[-1] // dof
    if rep == 'aa':
        pose = pose.reshape([-1, 3])
        pose_hat = pose_hat.reshape([-1, 3])
        ja = quaternion.from_rotation_vector(pose)
        ja_hat = quaternion.from_rotation_vector(pose_hat)
    else:
        pose = pose.reshape([-1, 3, 3])
        pose_hat = pose_hat.reshape([-1, 3, 3])
        ja = quaternion.from_rotation_matrix(pose)
        ja_hat = quaternion.from_rotation_matrix(pose_hat)
    angle_diff = quaternion.rotation_intrinsic_distance(ja, ja_hat)
    angle_diff = np.rad2deg(angle_diff).reshape(-1, n_joints)

    return angle_diff

