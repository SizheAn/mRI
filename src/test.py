
from aniposelib.cameras import CameraGroup
from scipy import optimize
import numpy as np
from read_cpl import get_videolabels, project2d
from scipy import signal
import time
from scipy.sparse import lil_matrix, dok_matrix
from camera_calibrate import getPmat


"""
Take in an array of 2D points of shape CxNxJx2,
an array of 3D points of shape NxJx3,
and an array of constraints of shape Kx2, where
C: number of camera
N: number of frames
J: number of joints
K: number of constraints

This function creates an optimized array of 3D points of shape NxJx3.

Example constraints:
constraints = [[0, 1], [1, 2], [2, 3]]
(meaning that lengths of segments 0->1, 1->2, 2->3 are all constant)

"""

camera_params = getPmat()
cgroup = CameraGroup.load('/disk/san49/github/anipose/calibration.toml')





subject = 'subject1'
dict, video_labels = get_videolabels(subject)


pose_2d_l = dict['pose_2d_l'].transpose(0,2,1)
pose_2d_r = dict['pose_2d_r'].transpose(0,2,1)

points_2d = np.stack((pose_2d_r, pose_2d_l))
points_3d_init = dict['pose_3d_tri'].transpose(0,2,1)



def reprojection_error(p3d, p2d, camera_params):

    pose_2d_l = p2d[0]
    pose_2d_r = p2d[1]

    error_r = pose_2d_r.transpose(0,2,1).reshape(1, -1, 2) - project2d(camera_params['cam1_proj_mat'], p3d.transpose(0,2,1)).reshape((1, -1, 2))
    error_l = pose_2d_l.transpose(0,2,1).reshape(1, -1, 2) - project2d(camera_params['cam2_proj_mat'], p3d.transpose(0,2,1)).reshape((1, -1, 2))

    error = np.stack((error_r, error_l)).squeeze()
    return error

def medfilt_data(values, size=15):
    padsize = size+5
    vpad = np.pad(values, (padsize, padsize), mode='reflect')
    vpadf = signal.medfilt(vpad, kernel_size=size)
    return vpadf[padsize:-padsize]

def initialize_params_triangulation(p3ds,
                                        constraints=[],
                                        constraints_weak=[]):
    joint_lengths = np.empty(len(constraints), dtype='float64')
    joint_lengths_weak = np.empty(len(constraints_weak), dtype='float64')

    for cix, (a, b) in enumerate(constraints):
        lengths = np.linalg.norm(p3ds[:, a] - p3ds[:, b], axis=1)
        joint_lengths[cix] = np.median(lengths)


    for cix, (a, b) in enumerate(constraints_weak):
        lengths = np.linalg.norm(p3ds[:, a] - p3ds[:, b], axis=1)
        joint_lengths_weak[cix] = np.median(lengths)

    all_lengths = np.hstack([joint_lengths, joint_lengths_weak])
    med = np.median(all_lengths)
    if med == 0:
        med = 1e-3

    mad = np.median(np.abs(all_lengths - med))

    joint_lengths[joint_lengths == 0] = med
    joint_lengths_weak[joint_lengths_weak == 0] = med
    joint_lengths[joint_lengths > med+mad*5] = med
    joint_lengths_weak[joint_lengths_weak > med+mad*5] = med

    return np.hstack([p3ds.ravel(), joint_lengths, joint_lengths_weak])

def jac_sparsity_triangulation(p2ds,
                                constraints=[],
                                constraints_weak=[],
                                n_deriv_smooth=1):
    n_cams, n_frames, n_joints, _ = p2ds.shape
    n_constraints = len(constraints)
    n_constraints_weak = len(constraints_weak)

    p2ds_flat = p2ds.reshape((n_cams, -1, 2))

    point_indices = np.zeros(p2ds_flat.shape, dtype='int32')
    for i in range(p2ds_flat.shape[1]):
        point_indices[:, i] = i

    point_indices_3d = np.arange(n_frames*n_joints)\
                            .reshape((n_frames, n_joints))

    good = ~np.isnan(p2ds_flat)
    n_errors_reproj = np.sum(good)
    n_errors_smooth = (n_frames-n_deriv_smooth) * n_joints * 3
    n_errors_lengths = n_constraints * n_frames
    n_errors_lengths_weak = n_constraints_weak * n_frames

    n_errors = n_errors_reproj + n_errors_smooth + \
        n_errors_lengths + n_errors_lengths_weak

    n_3d = n_frames*n_joints*3
    n_params = n_3d + n_constraints + n_constraints_weak

    point_indices_good = point_indices[good]

    A_sparse = dok_matrix((n_errors, n_params), dtype='int16')

    # constraints for reprojection errors
    ix_reproj = np.arange(n_errors_reproj)
    for k in range(3):
        A_sparse[ix_reproj, point_indices_good * 3 + k] = 1

    # sparse constraints for smoothness in time
    frames = np.arange(n_frames-n_deriv_smooth)
    for j in range(n_joints):
        for n in range(n_deriv_smooth+1):
            pa = point_indices_3d[frames, j]
            pb = point_indices_3d[frames+n, j]
            for k in range(3):
                A_sparse[n_errors_reproj + pa*3 + k, pb*3 + k] = 1

    ## -- strong constraints --
    # joint lengths should change with joint lengths errors
    start = n_errors_reproj + n_errors_smooth
    frames = np.arange(n_frames)
    for cix, (a, b) in enumerate(constraints):
        A_sparse[start + cix*n_frames + frames, n_3d+cix] = 1

    # points should change accordingly to match joint lengths too
    frames = np.arange(n_frames)
    for cix, (a, b) in enumerate(constraints):
        pa = point_indices_3d[frames, a]
        pb = point_indices_3d[frames, b]
        for k in range(3):
            A_sparse[start + cix*n_frames + frames, pa*3 + k] = 1
            A_sparse[start + cix*n_frames + frames, pb*3 + k] = 1

    ## -- weak constraints --
    # joint lengths should change with joint lengths errors
    start = n_errors_reproj + n_errors_smooth + n_errors_lengths
    frames = np.arange(n_frames)
    for cix, (a, b) in enumerate(constraints_weak):
        A_sparse[start + cix*n_frames + frames, n_3d + n_constraints + cix] = 1

    # points should change accordingly to match joint lengths too
    frames = np.arange(n_frames)
    for cix, (a, b) in enumerate(constraints_weak):
        pa = point_indices_3d[frames, a]
        pb = point_indices_3d[frames, b]
        for k in range(3):
            A_sparse[start + cix*n_frames + frames, pa*3 + k] = 1
            A_sparse[start + cix*n_frames + frames, pb*3 + k] = 1

    return A_sparse

def error_fun_triangulation(params, p2ds,
                                constraints=[],
                                constraints_weak=[],
                                scores=None,
                                scale_smooth=10000,
                                scale_length=1,
                                scale_length_weak=0.2,
                                reproj_error_threshold=100,
                                reproj_loss='soft_l1',
                                n_deriv_smooth=1):
    n_cams, n_frames, n_joints, _ = p2ds.shape

    n_3d = n_frames*n_joints*3
    n_constraints = len(constraints)
    n_constraints_weak = len(constraints_weak)

    # load params
    p3ds = params[:n_3d].reshape((n_frames, n_joints, 3))
    joint_lengths = np.array(params[n_3d:n_3d+n_constraints])
    joint_lengths_weak = np.array(params[n_3d+n_constraints:])

    # # reprojection errors
    # p3ds_flat = p3ds.reshape(-1, 3)
    errors_reproj = reprojection_error(p3ds, p2ds, camera_params)
    errors_reproj = errors_reproj.reshape(errors_reproj.shape[0]*errors_reproj.shape[1]*errors_reproj.shape[2])
    # if scores is not None:
    #     scores_flat = scores.reshape((n_cams, -1))
    #     errors = errors * scores_flat[:, :, None]
    # errors_reproj = errors[~np.isnan(p2ds_flat)]
    rp = reproj_error_threshold
    errors_reproj = np.abs(errors_reproj)
    if reproj_loss == 'huber':
        bad = errors_reproj > rp
        errors_reproj[bad] = rp*(2*np.sqrt(errors_reproj[bad]/rp) - 1)
    elif reproj_loss == 'linear':
        pass
    elif reproj_loss == 'soft_l1':
        errors_reproj = rp*2*(np.sqrt(1+errors_reproj/rp)-1)

    # temporal constraint
    errors_smooth = np.diff(p3ds, n=n_deriv_smooth, axis=0).ravel() * scale_smooth

    # joint length constraint
    errors_lengths = np.empty((n_constraints, n_frames), dtype='float64')
    for cix, (a, b) in enumerate(constraints):
        lengths = np.linalg.norm(p3ds[:, a] - p3ds[:, b], axis=1)
        expected = joint_lengths[cix]
        errors_lengths[cix] = 100*(lengths - expected)/expected
    errors_lengths = errors_lengths.ravel() * scale_length

    errors_lengths_weak = np.empty((n_constraints_weak, n_frames), dtype='float64')
    for cix, (a, b) in enumerate(constraints_weak):
        lengths = np.linalg.norm(p3ds[:, a] - p3ds[:, b], axis=1)
        expected = joint_lengths_weak[cix]
        errors_lengths_weak[cix] = 100*(lengths - expected)/expected
    errors_lengths_weak = errors_lengths_weak.ravel() * scale_length_weak

    # print("reproj error is: ", np.sum(errors_reproj))
    # print("smooth error is: ", np.sum(errors_smooth))
    # print("length error is: ", np.sum(errors_lengths))
    # print("length weak error is: ", np.sum(errors_lengths_weak))

    return np.hstack([errors_reproj, errors_smooth,
                        errors_lengths, errors_lengths_weak])

def optim_points(points, p3ds,
                    constraints=[],
                    constraints_weak=[],
                    scale_smooth=4,
                    scale_length=2, scale_length_weak=0.5,
                    reproj_error_threshold=15, reproj_loss='soft_l1',
                    n_deriv_smooth=1, scores=None, verbose=False):
    """
    Take in an array of 2D points of shape CxNxJx2,
    an array of 3D points of shape NxJx3,
    and an array of constraints of shape Kx2, where
    C: number of camera
    N: number of frames
    J: number of joints
    K: number of constraints

    This function creates an optimized array of 3D points of shape NxJx3.

    Example constraints:
    constraints = [[0, 1], [1, 2], [2, 3]]
    (meaning that lengths of segments 0->1, 1->2, 2->3 are all constant)

    """


    n_cams, n_frames, n_joints, _ = points.shape
    constraints = np.array(constraints)
    constraints_weak = np.array(constraints_weak)

    p3ds_med = np.apply_along_axis(medfilt_data, 0, p3ds, size=7)

    default_smooth = 1.0/np.mean(np.abs(np.diff(p3ds_med, axis=0)))
    scale_smooth_full = scale_smooth * default_smooth

    t1 = time.time()

    x0 = initialize_params_triangulation(
        p3ds, constraints, constraints_weak)

    x0[~np.isfinite(x0)] = 0

    jac = jac_sparsity_triangulation(
        points, constraints, constraints_weak, n_deriv_smooth)

    opt2 = optimize.least_squares(error_fun_triangulation,
                                    x0=x0, jac_sparsity=jac,
                                    loss='linear',
                                    ftol=1e-3,
                                    verbose=2*verbose,
                                    args=(points,
                                        constraints,
                                        constraints_weak,
                                        scores,
                                        scale_smooth_full,
                                        scale_length,
                                        scale_length_weak,
                                        reproj_error_threshold,
                                        reproj_loss,
                                        n_deriv_smooth))

    p3ds_new2 = opt2.x[:p3ds.size].reshape(p3ds.shape)

    t2 = time.time()

    if verbose:
        print('optimization took {:.2f} seconds'.format(t2 - t1))

    return p3ds_new2


config = {}
config['triangulation']  = {
    'ransac': False,
    'optim': False,
    'scale_smooth': 100,
    'scale_length': 2,
    'scale_length_weak': 1,
    'reproj_error_threshold': 100000,
    'score_threshold': 0.8,
    'n_deriv_smooth': 3,
    'constraints': [[5, 6],
                    [5, 7], [6, 8], [7, 9], [8, 10],
                    [5, 11], [6, 12],
                    [11,13], [12, 14], [13, 15], [14, 16]],
    'constraints_weak': [[0,5], [0,6]]
}

points_3d = optim_points(
    points_2d, points_3d_init,
    constraints=config['triangulation']['constraints'],
    constraints_weak=config['triangulation']['constraints_weak'],
    # scores=scores_2d,
    scale_smooth=config['triangulation']['scale_smooth'],
    scale_length=config['triangulation']['scale_length'],
    scale_length_weak=config['triangulation']['scale_length_weak'],
    n_deriv_smooth=config['triangulation']['n_deriv_smooth'],
    reproj_error_threshold=config['triangulation']['reproj_error_threshold'],
    verbose=True)

np.save('scipy_optim.npy', points_3d)
# points_3d = cgroup.optim_points(
#     points_2d, points_3d_init,
#     constraints=config['triangulation']['constraints'],
#     constraints_weak=config['triangulation']['constraints_weak'],
#     # scores=scores_2d,
#     scale_smooth=config['triangulation']['scale_smooth'],
#     scale_length=config['triangulation']['scale_length'],
#     scale_length_weak=config['triangulation']['scale_length_weak'],
#     n_deriv_smooth=config['triangulation']['n_deriv_smooth'],
#     reproj_error_threshold=config['triangulation']['reproj_error_threshold'],
#     verbose=True)



print("refined successfully")
