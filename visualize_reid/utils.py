import os.path as osp
import json
import open3d as o3d

import numpy as np
from scipy.spatial.transform import Rotation
import cv2


def unfold_camera_param(camera):
    world2color = np.linalg.inv(camera["color2world"])
    R, T = homogenous_to_rot_trans(world2color)
    camera["R"] = R
    camera["T"] = T
    fx = camera['fx']
    fy = camera['fy']
    # f = 0.5 * (camera['fx'] + camera['fy'])
    f = np.array([[fx], [fy]]).reshape(-1, 1)
    c = np.array([[camera['cx']], [camera['cy']]]).reshape(-1, 1)
    k = camera['k']
    p = camera['p']
    return R, T, f, c, k, p

def rot_trans_to_homogenous(rot, trans):
    """
    Args
        rot: 3x3 rotation matrix
        trans: 3x1 translation vector
    Returns
        4x4 homogenous matrix
    """
    X = np.zeros((4, 4))
    X[:3, :3] = rot
    X[:3, 3] = trans.T
    X[3, 3] = 1
    return X


def homogenous_to_rot_trans(X):
    """
    Args
        x: 4x4 homogenous matrix
    Returns
        rotation, translation: 3x3 rotation matrix, 3x1 translation vector
    """

    return X[:3, :3], X[:3, 3].reshape(3, 1)


def rotation_to_homogenous(vec):
    rot_mat = Rotation.from_rotvec(vec)
    swap = np.identity(4)
    swap = np.zeros((4, 4))
    swap[:3, :3] = rot_mat.as_matrix()
    swap[3, 3] = 1
    return swap


def load_camera_params(cam, dataset_root):
    scaling = 1000
    ds = {"id": cam}
    intrinsics = osp.join(dataset_root, cam, 'camera_calibration.yml')
    assert osp.exists(intrinsics)
    fs = cv2.FileStorage(intrinsics, cv2.FILE_STORAGE_READ)
    color_intrinsics = fs.getNode("undistorted_color_camera_matrix").mat()
    ds['fx'] = color_intrinsics[0, 0]
    ds['fy'] = color_intrinsics[1, 1]
    ds['cx'] = color_intrinsics[0, 2]
    ds['cy'] = color_intrinsics[1, 2]
    # images are undistorted! Just put 0. Voxelpose assumes just 4 dist coeffs
    dist = fs.getNode("color_distortion_coefficients").mat()
    #ds['k'] = np.array(dist[[0, 1, 4, 5, 6, 7]])
    #ds['p'] = np.array(dist[2:4])
    ds['k'] = np.zeros((3, 1))
    ds['p'] = np.zeros((2, 1))

    depth2color_r = fs.getNode('depth2color_rotation').mat()
    # depth2color_t is in mm by default, change all to meters
    depth2color_t = fs.getNode('depth2color_translation').mat() / scaling

    depth2color = rot_trans_to_homogenous(depth2color_r, depth2color_t.reshape(3))
    ds["depth2color"] = depth2color

    extrinsics = osp.join(dataset_root, cam, "world2camera.json")
    with open(extrinsics, 'r') as f:
        ext = json.load(f)
        trans = np.array([x for x in ext['translation'].values()])
        # NOTE: world2camera translation convention is in meters. Here we convert
        # to mm. Seems like Voxelpose was using mm as well.
        # trans = trans * 1000
        _R = ext['rotation']
        rot = Rotation.from_quat([_R['x'], _R['y'], _R['z'], _R['w']]).as_matrix()
        ext_homo = rot_trans_to_homogenous(rot, trans)
        # flip coordinate transform back to opencv convention

    yz_flip = rotation_to_homogenous(np.pi * np.array([1, 0, 0]))
    YZ_SWAP = rotation_to_homogenous(np.pi/2 * np.array([1, 0, 0]))

    # first swap into OPENGL convention, then we can apply intrinsics.
    # then swap into our own Z-up prefered format..
    depth2world = YZ_SWAP @ ext_homo @ yz_flip
    # print(f"{cam} extrinsics:", depth2world)

    depth_R, depth_T = homogenous_to_rot_trans(depth2world)
    ds["depth2world"] = depth2world
    color2world = depth2world @ np.linalg.inv(depth2color)

    R, T = homogenous_to_rot_trans(color2world)
    ds["color2world"] = color2world
    return ds


def project_points_radial(x, R, T, K, k, p):
    """
    Args
        x: Nx3 points in world coordinates R: 3x3 Camera rotation matrix
        T: 3x1 Camera translation parameters
        K: 3x3 Camera intrinsic matrix
        k: 3x1 Camera radial distortion coefficients
        p: 2x1 Camera tangential distortion coefficients
    Returns
        ypixel.T: Nx2 points in pixel space
    """
    n = x.shape[0]
    x = x
    # x = np.multiply([-1, 1, 1], x)
    # world2camera
    # https://www-users.cs.umn.edu/~hspark/CSci5980/Lec2_ProjectionMatrix.pdf
    xcam = R.dot(x.T) + T
    xcam = K @ xcam

    ypixel = xcam[:2] / (xcam[2]+1e-5)
    # print(xcam[2])

    # r2 = np.sum(y**2, axis=0)
    # radial = 1 + np.einsum('ij,ij->j', np.tile(k, (1, n)),
    #                        np.array([r2, r2**2, r2**3]))
    # tan = p[0] * y[1] + p[1] * y[0]
    # y = y * np.tile(radial + 2 * tan,
    #                 (2, 1)) + np.outer(np.array([p[1], p[0]]).reshape(-1), r2)
    # ypixel = np.multiply(f, y) + c
    return ypixel.T


def project_points_opencv(x, R, T, K, k, p):
    dist_coefs = np.concatenate([k[0:2].T[0], p.T[0], k[2:].T[0]])
    # rvec, T perform a change of basis from world to camera coordinate system
    rvec = cv2.Rodrigues(R)[0]
    # project from 3D to 2D. projectPoints handles rotation and translation
    points_2d = cv2.projectPoints(x, rvec, T, K, dist_coefs)
    # TODO: why does projectPoints nest arrays like this?
    return np.array([x[0] for x in points_2d[0]])


def project_pose(x, camera):
    R, T, f, c, k, p = unfold_camera_param(camera)

    K = np.array([f[0][0], 0, c[0][0],
                  0, f[1][0], c[1][0],
                  0, 0, 1])
    K = K.reshape(3,3)
    loc2d_opencv = project_points_opencv(x, R, T, K, k, p)
    loc2d = project_points_radial(x, R, T, K, k, p)

    # print(camera["id"])
    # print("------------")
    # print(f" loc2d -> {loc2d} \n opencv -> {loc2d_opencv}")
    return loc2d