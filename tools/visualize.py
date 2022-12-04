import os
from copy import deepcopy

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import cv2

from visualize_reID.utils import load_camera_params
from visualize_reID.utils import project_pose, homogenous_to_rot_trans

DATA_DIR = "/media/ana/Multimedia/UbuntuChestii/bodytracking"
CAMERAS = ["cn01", "cn02", "cn03", "cn04", "cn05", "cn06"]


def project_to_views(point, frame_id):
    # cam 1 [750, 640]
    projected_points = []
    proj_pts2 = dict.fromkeys(["cn01", "cn02", "cn03", "cn04", "cn05", "cn06"])
    for i, cam in enumerate(CAMERAS[:]):
        file_id = str(frame_id).zfill(10)
        params = load_camera_params(cam, DATA_DIR)
        loc2d = project_pose(point, params)[0]
        fpath = os.path.join(DATA_DIR, cam, f"{file_id}_color.jpg")
        color = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
        if color is None:
            print("File not found: ", fpath)
        # shape is in form: [height, width, channel]
        height, width, _ = color.shape
        # y-val <-> height x-val <-> width
        kinect_offset = np.array([0.5, 0.5])
        x, y = np.int16(np.round(loc2d - kinect_offset))
        # print('*' * 30)
        # print(x, y)
        # print('*' * 30)
        if 0 < x < width and 0 < y < height:
            # print("Point present in image")
            # print("\n")
            # azure kinect uses reverse indexing
            cv2.circle(color, (x, y), 5, (255, 255, 255), 5)
            cv2.imwrite(cam + "_test.jpg", color)
            projected_points.append([x,y])
            proj_pts2[cam] = [x,y]

    # return projected_points
    return proj_pts2

def add_mesh_sphere(point, vis, name):
    # add a placeholder sphere
    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
    mesh_sphere.compute_vertex_normals()
    mesh_sphere.paint_uniform_color([0.7, 0.7, 0.1])
    mesh_sphere.translate(point)
    vis.add_geometry(name, mesh_sphere)


def render_camera_poses(points, vis, frame_id, multiple_pts=False):
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[0, 0, 0])
    vis.add_geometry("coordinate_frame", mesh_frame)
    for cam in CAMERAS:
        file_id = str(frame_id).zfill(4)
        fpath = os.path.join(DATA_DIR, cam, f"{file_id}_pointcloud.ply")
        if not os.path.exists(fpath):
            print("File does not exist: ", fpath)
            continue
        ply = o3d.io.read_point_cloud(fpath)
        params = load_camera_params(cam, DATA_DIR)
        # inv = np.linalg.inv(params["depth2world"])
        # ply.transform(inv)
        vis.add_geometry(f"{cam}-ply", ply)

        print(f"Transforming for camera {cam}")
        camera_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
        # world2color = params["depth2color"]
        world2color = params["color2world"]
        camera_origin.transform(world2color)
        print(camera_origin.get_center())
        vis.add_geometry(f"{cam}-color", camera_origin)
        vis.add_3d_label(camera_origin.get_center(), cam)

        depth_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
        world2depth = params["depth2world"]
        depth_origin.transform(world2depth)
        print(depth_origin.get_center())
        vis.add_geometry(f"{cam}-depth-cam", depth_origin)
        vis.add_3d_label(depth_origin.get_center(), cam + 'depth-')

    if multiple_pts:
        for i, point in enumerate(points):
            add_mesh_sphere(point, vis, f"sphere{i}")
    else:
        add_mesh_sphere(points, vis, "sphere")


def render_single_transform(point, vis):
    cam = "cn02"
    file_id = str(frame_id).zfill(4)
    ply = o3d.io.read_point_cloud(os.path.join(DATA_DIR, cam, f"{file_id}_pointcloud.ply"))
    vis.add_geometry(f"{cam}-ply-camera", ply)
    params = load_camera_params(cam, DATA_DIR)
    depth2world = params["depth2world"]
    R, T = homogenous_to_rot_trans(depth2world)
    add_mesh_sphere(R.T @ (point.T - T), vis, 'sphere-cam')
    # add the pointcloud in world coordinate system
    ply_world = deepcopy(ply)
    ply_world.transform(depth2world)
    vis.add_geometry(f"{cam}-ply-world", ply_world)
    add_mesh_sphere(point, vis, 'sphere-world')


if __name__ == "__main__":
    # draw ball at point
    frame_id = 1100
    np.set_printoptions(suppress=True)
    #point = np.array([0.265301, -0.963982, 0.005958])
    #point = np.array([0.38056242, -0.75900618,  0.15178271])
    #point = np.array([-0.32447208, -0.7205877,  -0.75082322])
    #point = np.array([0.48097092, -1.23508413, -0.27610298])

    # point = np.array([1.90920418, 1.84715575, 1.95422496])  # depth/100; cn01; id:0; frame:1100; promising
    # point = np.array([2.10510641, 2.29084316, 2.2879746])  # depth/1000; cn01; id:0; frame:1100; too far

    # point = np.array([2.06128664, 2.24673001, 2.25590372])  # depth/100; cn01; id:0; frame:1015; too far
    # point = np.array([1.47100648, 1.40602423, 1.63351623])  # depth/10; cn01; id:0; frame:1015; promising

    # point = np.array([1.57148915, 1.50996333, 1.71104427])  # depth/100; cn01; id:0; frame:1020; promising
    # point = np.array([1.57283952, 1.51825849, 1.70325816])  # depth/100; cn01; id:0; frame:1025; promising

    # points = np.array([[1.10379721, 0.8108657 , 1.19398018], [0.35765213, -0.20135986,  1.43146217]])
    points = np.array([1.27436755, 0.55804525, 1.02190033])
    # new extrinsics
    #point = np.array([0.249721, -0.005661, -0.974014])
    app = gui.Application.instance
    app.initialize()
    vis = o3d.visualization.O3DVisualizer("Open3D - 3D Text", 1024, 768)
    vis.show_settings = True
    render_camera_poses(points, vis, frame_id)
    #render_single_transform(point, vis)
    #project_to_views(point.reshape(1, 3), frame_id)
    app.add_window(vis)
    app.run()