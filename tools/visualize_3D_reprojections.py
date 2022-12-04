import os
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import cv2
from visualize_reID.visualize import render_camera_poses
from visualize_reID.utils import load_camera_params, unfold_camera_param

DATA_DIR = "/media/ana/Multimedia/UbuntuChestii/bodytracking/"
CAMERAS = ["cn01", "cn02", "cn03", "cn04", "cn05", "cn06"]
center_tracks = list()
ids = list()
MAX_ID: int = 0

# invert the camera parameters for the reprojection into 3D
def calculate_reprojection_params(params):
    # extrinsic parameters: R = rotation matrix, T = translation vector
    # intrinsic parameters: f = focal length, c = principal point, k, p = distortion coefficients
    R, T, f, c, k, p = unfold_camera_param(params)
    K = np.array([f[0][0], 0, c[0][0],
                  0, f[1][0], c[1][0],
                  0, 0, 1])

    K = K.reshape(3,3)
    K_inv = np.linalg.inv(K)
    R_t = R.T
    T_inv = (-R_t @ T).T
    return K_inv, R_t, T_inv


# convert a pixel coordinate into homogenous coordinates
def convert_to_hom_coords(point):
    point = np.append(point, 1)
    return point

# reproject one pixel into the corresponding 3D point
def reproject_pixel_in_3D(camera, px_coords, frame_id):
    # convert the pixel into homogenous coordinates
    px_coords = convert_to_hom_coords(px_coords)
    # read the depth mask
    file_id = str(frame_id).zfill(10)
    fpath = os.path.join(DATA_DIR, camera, f"{file_id}_rgbd.tiff")
    depth_mask = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
    # depth_mask is flipped
    # a pixel (x,y) in the color image can be accessed by (y,x) in the depth mask
    depth = depth_mask[px_coords[1]][px_coords[0]]/1000
    # the field of view of the depth camera is smaller than the one for the rgb images
    # need to check whether we have a measurement for the given pixel
    if depth == 0.0:
        print(f"for camera:{camera} the pixel:{px_coords[0], px_coords[1]} has depth=0.0; cannot perform reprojection")
        return None
    # load the camera parameters (extrinsics and intrinsics)
    # they perform the mapping from a world point in 3D to a pixel in 2D
    params = load_camera_params(camera, DATA_DIR)
    # invert the camera parameters to get a reprojection from 2D into 3D
    K_inv, R_t, T_inv = calculate_reprojection_params(params)
    px_to_depth_cam = K_inv @ px_coords * depth
    depth_cam_to_world = R_t @ px_to_depth_cam + T_inv

    return depth_cam_to_world


# construct the 3D reprojections for each pixel
def construct_world_points(pixels, frame_id):
    world_points = list()
    for cam in CAMERAS[:]:
        for pixel in pixels[cam]:
            point = reproject_pixel_in_3D(cam, pixel, frame_id)
            if point is not None:
                world_points.append(point.reshape(3,))
    return world_points


# draw circles on the position of the pixels for a better intuition on where the 3D reprojections should end up
def draw_centers():
    for cam in CAMERAS[:]:
        file_id = str(frame_id).zfill(10)
        fpath = os.path.join(DATA_DIR, cam, f"{file_id}_color.jpg")
        image = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
        if image is None:
            print("File not found: ", fpath)
        cv2.circle(image, (pixels[cam][0], pixels[cam][1]), 5, (255, 255, 255), 5)
        cv2.imwrite(cam + "_frame_1100_center.jpg", image)


def fuse(world_points):
    if len(world_points) == 0:
        print("error: there are no world points to fuse!")
        return None

    if len(world_points) == 1:
        return world_points[0]

    sum = np.sum(world_points, axis=0)
    fused_point = sum / len(world_points)
    return fused_point


def closest_center(point):
    human_width = 0.57
    for pos, center in enumerate(center_tracks):
        print(np.linalg.norm(point-center))
        if np.linalg.norm(point-center) <= human_width:
            return pos
    return None


def init():
    d = {}
    for i in range(len(center_tracks)):
        d[i] = list()
    return d


def fuse_world_points(world_points):
    global MAX_ID
    if not center_tracks:
        center_tracks.append(world_points[0])
        ids.append(MAX_ID)
        MAX_ID += 1
        # fused = fuse(world_points)
        # center_tracks.append(fused)
        # ids.append(MAX_ID)
        # MAX_ID += 1
        # return fused

    to_fuse : dict = init()
    for point in world_points:
        pos_closest_center = closest_center(point)
        if pos_closest_center is not None:
            print("*")
            to_fuse[pos_closest_center].append(point)
        else:
            print("/")
            to_fuse[MAX_ID] = list()
            to_fuse[MAX_ID].append(point)
            center_tracks.append(point)
            ids.append(MAX_ID)
            MAX_ID += 1

    for key in to_fuse.keys():
        if key < len(center_tracks):
            fused_pt = fuse(to_fuse[key])
            if fused_pt is not None:
                center_tracks[key] = fused_pt
            else: # someone disappears
                ids.remove(key)
                print(f"id to remove: {key}")
        else: # someone appears
            center_tracks.append(to_fuse[key])

    return center_tracks, ids


def visualize(points, frame_id, multiplePoints=False):
    app = gui.Application.instance
    app.initialize()
    vis = o3d.visualization.O3DVisualizer("Open3D - 3D Text", 1024, 768)
    vis.show_settings = True
    render_camera_poses(points, vis, frame_id, multiple_pts=multiplePoints)
    app.add_window(vis)
    app.run()


if __name__ == "__main__":

    frame_id = 1755
    # center pixels of a person in the frame 1755
    pixels = {"cn01": np.array([[1024, 417], [563, 475], [248, 553]]),
              # "cn03": np.array([[342, 1080], [759, 1104]]), #
              "cn04": np.array([[925, 773], [1548, 848]]),
              "cn05": np.array([[1346, 974], [1224, 601], [1180, 507]]),
              "cn06": np.array([[453, 698]])
              }
    # visualize the centers on the 2D images
    # draw_centers()
    # reproject the pixels in 3D
    world_pts = np.array(construct_world_points(pixels, frame_id))
    visualize(world_pts, frame_id, multiplePoints=True)
    fused_pt, ids = fuse_world_points(world_pts)
    print(fused_pt)
    visualize(fused_pt, frame_id, multiplePoints=True)

    frame_id = 1760
    # center pixels of a person in the frame 1105
    pixels = {"cn01": np.array([[1042, 419], [567, 481], [261, 560]]),
              # "cn02": np.array([[567, 481], [1003, 607], [1193, 1082]]),
              # "cn03": np.array([[351, 1055], [916, 1108]]), #
              "cn04": np.array([[906, 758], [1541, 862]]),
              "cn05": np.array([[1354, 979], [1198, 581], [1153, 502]]),
              "cn06": np.array([[482, 691]])
              }
    # reproject the pixels in 3D
    world_pts = np.array(construct_world_points(pixels, frame_id))
    visualize(world_pts, frame_id, multiplePoints=True)
    fused_pts, ids = fuse_world_points(world_pts)
    print(fused_pts)
    visualize(fused_pts, frame_id, multiplePoints=True)


    # frame_id = 1100
    # # center pixels of a person in the frame 1100
    # pixels = {"cn01": np.array([680, 718]),
    #           "cn02": np.array([1695, 637]),
    #           "cn03": np.array([1692, 496]),
    #           "cn04": np.array([1473, 540])
    #           }
    # # visualize the centers on the 2D images
    # draw_centers()
    # # reproject the pixels in 3D
    # world_pts = np.array(construct_world_points(pixels))
    # # visualize(world_pts, frame_id, multiplePoints=True)
    # fused_pt = fuse_world_points(world_pts)
    # print(fused_pt)
    # # visualize(fused_pt, frame_id)
    #
    # frame_id = 1105
    # # center pixels of a person in the frame 1105
    # pixels = {"cn01": np.array([676, 713]),
    #           "cn02": np.array([1705, 639]),
    #           "cn03": np.array([1705, 492]),
    #           "cn04": np.array([1474, 537])
    #           }
    # # reproject the pixels in 3D
    # world_pts = np.array(construct_world_points(pixels))
    # # visualize(world_pts, frame_id, multiplePoints=True)
    # fused_pts = fuse_world_points(world_pts)
    # print(fused_pts)
    # # visualize(fused_pts, frame_id, multiplePoints=True)
