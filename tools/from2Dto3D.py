import os
from typing import Optional, Any

import numpy as np
#import open3d as o3d
#import open3d.visualization.gui as gui
import cv2

from visualize_reID.utils import load_camera_params
from visualize_reID.utils import unfold_camera_param
from visualize_reID.visualize import project_to_views

#DATA_DIR = "D:\Doc\Desktop\\bodytracking"
DATA_DIR = "/home/ana/Downloads/bodytracking"
# DATA_DIR = "/home/ana/Downloads/debug_output"
CAMERAS = ["cn01", "cn02", "cn04", "cn05", "cn06"]


def calculate_inverse_projection_params(params):
    R, T, f, c, k, p = unfold_camera_param(params)
    K = np.array([f[0][0], 0, c[0][0],
                  0, f[1][0], c[1][0],
                  0, 0, 1])

    K = K.reshape(3,3)
    K_inv = np.linalg.inv(K)
    R_t = R.T
    T_inv = (-R_t @ T).T
    return K_inv, R_t, T_inv


# def convert_to_hom_coords(px_coords):
#     for point in px_coords:
#         point.append(1)
#     px_coords = np.array(px_coords)
#     return px_coords


def convert_to_hom_coords(point):
    point.append(1)
    return point


def project_pixel_in_3D(cameras, px_coords, frame_id):
    world_points = list()
    for i, cam in enumerate(cameras):
        file_id = str(frame_id).zfill(10)
        fpath = os.path.join(DATA_DIR, cam, f"{file_id}_rgbd.tiff")
        depth_mask = cv2.imread(fpath,
                          cv2.IMREAD_UNCHANGED)
        # sliced = depth_mask[1212:1525, 268:835]
        # sliced = depth_mask[675:970, 285:976]
        # sliced_filtered = sliced[sliced > 0]
        # av_depth = np.average(sliced)
        # #sliced2 = sliced[:, 268:835]

        norm_img = np.zeros((1536, 2048))
        if px_coords[i][1] > 1536 or px_coords[i][0] > 2048:
            continue
        depth_mask_norm = cv2.normalize(depth_mask, norm_img, 0, 255, cv2.NORM_MINMAX)
        # cv2.circle(depth_mask_norm, (px_coords[i][0], px_coords[i][1]), 5, (255, 255, 255), 5)
        # cv2.imshow("", depth_mask_norm)
        #cv2.waitKey()
        cv2.imwrite(f"/home/ana/Downloads/bodytracking/depth_{cam}.tiff", depth_mask_norm)

        depth = depth_mask[px_coords[i][1]][px_coords[i][0]]
        print(depth)
        depth = depth/1000
        params = load_camera_params(cam, DATA_DIR)
        K_inv, R_t, T_inv = calculate_inverse_projection_params(params)
        px_to_cam = K_inv @ px_coords[i] * depth
        cam_to_world = R_t @ px_to_cam + T_inv

        world_points.append(cam_to_world)

    return world_points
    # return cam_to_world

# def project_pixel_in_3D(px_coords, frame_id):
#     world_points = list()
#     for i, cam in enumerate(CAMERAS[:]):
#         file_id = str(frame_id).zfill(10)
#         fpath = os.path.join(DATA_DIR, cam, f"{file_id}_rgbd.tiff")
#         depth_mask = cv2.imread(fpath,
#                           cv2.IMREAD_GRAYSCALE)
#         depth = depth_mask[px_coords[i][1]][px_coords[i][0]]
#         #depth -= 1e-5
#         params = load_camera_params(cam, DATA_DIR)
#         K_inv, R_t, T_inv = calculate_inverse_projection_params(params)
#         px_to_cam = K_inv @ px_coords[i] * depth
#         cam_to_world = R_t @ px_to_cam + T_inv
#         world_points.append(cam_to_world)
#
#     return world_points


def fuse_world_points(world_pts):
    # TODO
    point = np.array([0.265301, -0.963982, 0.005958])
    return point.reshape(1,3)


def id_is_present(id, element):
    if element is None:
        return None

    for i in range(len(element)):
        if element[i][0] == id:
            return i

    return None


def reinit(information, cam):
    if not information[cam]:
        information[cam] = None


def update_information(id, information, optimized_2D_centers):
    for cam, optimized_center in optimized_2D_centers.items():
        person_index = id_is_present(id, information[cam])
        if person_index is not None:
            if optimized_center is None:
                del information[cam][person_index]
                reinit(information, cam)
            else:
                information[cam][person_index] = (information[cam][person_index][0], optimized_center, information[cam][person_index][2])
        else:
            if optimized_center is not None:
                information[cam].append((id, optimized_center, [(0, 0), (0, 0)]))
#   return information


def retrieve_center(id, element):
    if element is None:
        return None
    for i in range(len(element)):
        if element[i][0] == id:
            return element[i][1]
    return None


def optimize_detection_centers(information, global_ids, frame_id):
    for i in global_ids:
        centers = []
        cameras = []
        for cam, element in information.items():
            center = retrieve_center(global_ids[i], element)
            if center is not None:
                centers.append(convert_to_hom_coords(center))
                cameras.append(cam)
        centers = np.array(centers)
        world_pts = project_pixel_in_3D(cameras, centers, frame_id)
        optimized_center = fuse_world_points(world_pts)
        optimized_2D_centers = project_to_views(optimized_center, frame_id)
        update_information(global_ids[i], information, optimized_2D_centers)
#    return information


if __name__ == "__main__":
    frame_id = 1100
    pxs = np.array([[752, 640, 1], [871, 908, 1], [1320, 1245, 1], [1281, 899, 1], [104, 1318, 1]])
    world_pts = project_pixel_in_3D(CAMERAS[:], pxs, frame_id)
    print("reconstructed 3d pts")
    for i, cam in enumerate(CAMERAS[:]):
        print(cam, end=": ")
        print(world_pts[i])

    actual_pt = np.array([0.265301, -0.963982, 0.005958])
    print("actual world point")
    print(actual_pt)
