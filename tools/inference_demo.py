# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# The code is based on deep-high-resolution-net.pytorch.
# (https://github.com/leoxiaobin/deep-high-resolution-net.pytorch)
# Modified by Zigang Geng (zigang@mail.ustc.edu.cn).
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import csv
import os
import shutil
import time
import sys
sys.path.append("../lib")

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision

import _init_paths
import models

from config import cfg
from config import update_config
from core.inference import get_multi_stage_outputs
from core.inference import aggregate_results
from core.nms import pose_nms
from core.match import match_pose_to_heatmap
from utils.transforms import resize_align_multi_scale
from utils.transforms import get_final_preds
from utils.transforms import get_multi_scale_size
from utils.transforms import up_interpolate


from visualize_reID.visualize_3D_reprojections import construct_world_points, visualize, fuse_world_points, reproject_pixel_in_3D

CTX = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


COCO_KEYPOINT_INDEXES = {
    0: 'nose',
    1: 'left_eye',
    2: 'right_eye',
    3: 'left_ear',
    4: 'right_ear',
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle'
}


CROWDPOSE_KEYPOINT_INDEXES = {
    0: 'left_shoulder',
    1: 'right_shoulder',
    2: 'left_elbow',
    3: 'right_elbow',
    4: 'left_wrist',
    5: 'right_wrist',
    6: 'left_hip',
    7: 'right_hip',
    8: 'left_knee',
    9: 'right_knee',
    10: 'left_ankle',
    11: 'right_ankle',
    12: 'head',
    13: 'neck'
}


def get_pose_estimation_prediction(cfg, model, image, vis_thre, transforms):
    # size at scale 1.0
    base_size, center, scale = get_multi_scale_size(
        image, cfg.DATASET.INPUT_SIZE, 1.0, 1.0
    )

    with torch.no_grad():
        heatmap_sum = 0
        poses = []

        for scale in sorted(cfg.TEST.SCALE_FACTOR, reverse=True):
            image_resized, center, scale_resized = resize_align_multi_scale(
                image, cfg.DATASET.INPUT_SIZE, scale, 1.0
            )

            image_resized = transforms(image_resized)
            image_resized = image_resized.unsqueeze(0).cuda()

            heatmap, posemap = get_multi_stage_outputs(
                cfg, model, image_resized, cfg.TEST.FLIP_TEST
            )
            heatmap_sum, poses = aggregate_results(
                cfg, heatmap_sum, poses, heatmap, posemap, scale
            )
        
        heatmap_avg = heatmap_sum/len(cfg.TEST.SCALE_FACTOR)
        poses, scores = pose_nms(cfg, heatmap_avg, poses)

        if len(scores) == 0:
            return []
        else:
            if cfg.TEST.MATCH_HMP:
                poses = match_pose_to_heatmap(cfg, poses, heatmap_avg)

            final_poses = get_final_preds(
                poses, center, scale_resized, base_size
            )

        final_results = []
        for i in range(len(scores)):
            if scores[i] > vis_thre:
                final_results.append(final_poses[i])

        if len(final_results) == 0:
            return []

    return final_results


def prepare_output_dirs(prefix='/output/'):
    pose_dir = os.path.join(prefix, "pose")
    if os.path.exists(pose_dir) and os.path.isdir(pose_dir):
        shutil.rmtree(pose_dir)
    os.makedirs(pose_dir, exist_ok=True)
    return pose_dir


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--videoFile', type=str, required=True)
    parser.add_argument('--outputDir', type=str, default='/output/')
    parser.add_argument('--inferenceFps', type=int, default=10)
    parser.add_argument('--visthre', type=float, default=0)
    parser.add_argument('opts',
                        help='Modify config options using the command-line',
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    # args expected by supporting codebase
    args.modelDir = ''
    args.logDir = ''
    args.dataDir = ''
    args.prevModelDir = ''
    return args

POSE_DIR = "/home/ana/PycharmProjects/DEKR/output/pose/"


def main(image_path, camera):
    # transformation
    pose_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    args = parse_args()
    update_config(cfg, args)
    # pose_dir = prepare_output_dirs(args.outputDir)
    pose_dir = POSE_DIR
    csv_output_rows = []

    pose_model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )

    if cfg.TEST.MODEL_FILE:
        print('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        pose_model.load_state_dict(torch.load(
            cfg.TEST.MODEL_FILE), strict=False)
    else:
        raise ValueError('expected model defined in config at TEST.MODEL_FILE')

    pose_model.to(CTX)
    pose_model.eval()

    # image_path = "/home/ana/Downloads/bodytracking/cn06/0000001755_color.jpg"
    image_bgr = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    image_debug = image_bgr.copy()
    image_rgb = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image_pose = image_rgb.copy()

    pose_preds = get_pose_estimation_prediction(
                 cfg, pose_model, image_pose, args.visthre, transforms=pose_transform)
    #new_csv_row = []
    for coords in pose_preds:
    # Draw each point on image
        new_csv_row = []
        for coord in coords:
            x_coord, y_coord = int(coord[0]), int(coord[1])
            cv2.circle(image_debug, (x_coord, y_coord), 4, (255, 0, 0), 2)
            new_csv_row.extend([x_coord, y_coord])
        csv_output_rows.append(new_csv_row)
        cv2.putText(image_debug, "", (100, 50), cv2.FONT_HERSHEY_SIMPLEX,
                   1, (0, 0, 255), 2, cv2.LINE_AA)

    # csv_output_rows.append(new_csv_row)
    # csv_output_rows.append(all_csv_rows)
    new_path = os.path.join(image_path.split('/')[-2],image_path.split('/')[-1])
    img_file = os.path.join(pose_dir, new_path)
    cv2.imwrite(img_file, image_debug)
    # outcap.write(image_debug)

    # write csv
    # csv_headers = ['frame']
    csv_headers = []
    if cfg.DATASET.DATASET_TEST == 'coco':
        for keypoint in COCO_KEYPOINT_INDEXES.values():
            csv_headers.extend([keypoint+'_x', keypoint+'_y'])
    # elif cfg.DATASET.DATASET_TEST == 'crowd_pose':
    #     for keypoint in CROWDPOSE_KEYPOINT_INDEXES.values():
    #         csv_headers.extend([keypoint+'_x', keypoint+'_y'])
    else:
        raise ValueError('Please implement keypoint_index for new dataset: %s.' % cfg.DATASET.DATASET_TEST)

    csv_output_filename = os.path.join(args.outputDir, f'pose-data-{camera}.csv')
    with open(csv_output_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(csv_headers)
        csvwriter.writerows(csv_output_rows)

    # vidcap.release()
    # outcap.release()

    cv2.destroyAllWindows()

    return img_file


def old_main():
    # transformation
    pose_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    args = parse_args()
    update_config(cfg, args)
    pose_dir = prepare_output_dirs(args.outputDir)
    csv_output_rows = []

    pose_model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )

    if cfg.TEST.MODEL_FILE:
        print('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        pose_model.load_state_dict(torch.load(
            cfg.TEST.MODEL_FILE), strict=False)
    else:
        raise ValueError('expected model defined in config at TEST.MODEL_FILE')

    pose_model.to(CTX)
    pose_model.eval()

    # Loading an video
    vidcap = cv2.VideoCapture(args.videoFile)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    if fps < args.inferenceFps:
        raise ValueError('desired inference fps is ' +
                         str(args.inferenceFps)+' but video fps is '+str(fps))
    print(f"fps=",{fps})
    print(f"argsFps=",{args.inferenceFps})
    skip_frame_cnt = round(fps / args.inferenceFps / 2)
    print(f"skip_frame_cnt=",{skip_frame_cnt})
    frame_width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    outcap = cv2.VideoWriter('{}/{}_pose.avi'.format(args.outputDir, os.path.splitext(os.path.basename(args.videoFile))[0]),
                             cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), int(skip_frame_cnt), (frame_width, frame_height))

    count = 0
    while vidcap.isOpened():
        total_now = time.time()
        ret, image_bgr = vidcap.read()
        count += 1

        if not ret:
            break

        if count % skip_frame_cnt != 0:
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        image_pose = image_rgb.copy()

        # Clone 1 image for debugging purpose
        image_debug = image_bgr.copy()

        now = time.time()
        pose_preds = get_pose_estimation_prediction(
            cfg, pose_model, image_pose, args.visthre, transforms=pose_transform)
        then = time.time()
        if len(pose_preds) == 0:
            count += 1
            continue

        print("Find person pose in: {} sec".format(then - now))

        # new_csv_row = []
        for coords in pose_preds:
            # Draw each point on image
            new_csv_row = []
            for coord in coords:
                x_coord, y_coord = int(coord[0]), int(coord[1])
                cv2.circle(image_debug, (x_coord, y_coord), 4, (255, 0, 0), 2)
                new_csv_row.extend([x_coord, y_coord])

        total_then = time.time()
        text = "{:03.2f} sec".format(total_then - total_now)
        cv2.putText(image_debug, text, (100, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2, cv2.LINE_AA)

        csv_output_rows.append(new_csv_row)
        img_file = os.path.join(pose_dir, 'pose_{:08d}.jpg'.format(count))
        cv2.imwrite(img_file, image_debug)
        outcap.write(image_debug)

    # write csv
    # csv_headers=['frame']
    csv_headers = []
    if cfg.DATASET.DATASET_TEST == 'coco':
        for keypoint in COCO_KEYPOINT_INDEXES.values():
            csv_headers.extend([keypoint+'_x', keypoint+'_y'])
    elif cfg.DATASET.DATASET_TEST == 'crowd_pose':
        for keypoint in CROWDPOSE_KEYPOINT_INDEXES.values():
            csv_headers.extend([keypoint+'_x', keypoint+'_y'])
    else:
        raise ValueError('Please implement keypoint_index for new dataset: %s.' % cfg.DATASET.DATASET_TEST)

    csv_output_filename = os.path.join(args.outputDir, 'pose-data.csv')
    with open(csv_output_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(csv_headers)
        csvwriter.writerows(csv_output_rows)

    vidcap.release()
    outcap.release()

    cv2.destroyAllWindows()


def extract_frames():

    args = parse_args()
    update_config(cfg, args)
    pose_dir = prepare_output_dirs(args.outputDir)

    # Loading an video
    vidcap = cv2.VideoCapture(args.videoFile)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    if fps < args.inferenceFps:
        raise ValueError('desired inference fps is ' +
                         str(args.inferenceFps)+' but video fps is '+str(fps))
    print(f"fps=",{fps})
    print(f"argsFps=",{args.inferenceFps})
    skip_frame_cnt = round(fps / args.inferenceFps / 2)
    print(f"skip_frame_cnt=",{skip_frame_cnt})

    count = 0
    while vidcap.isOpened():
        total_now = time.time()
        ret, image_bgr = vidcap.read()
        count += 1

        if not ret:
            break

        if count % skip_frame_cnt != 0:
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        image_pose = image_rgb.copy()

        # Clone 1 image for debugging purpose
        image_debug = image_bgr.copy()

        now = time.time()

        cv2.putText(image_debug, "", (100, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2, cv2.LINE_AA)

        img_file = os.path.join(pose_dir, 'pose_{:08d}.jpg'.format(count))
        cv2.imwrite(img_file, image_debug)

    vidcap.release()


def compute_center(key_points):
    separate_x = [int(x) for x in key_points[::2]]
    separate_y = [int(y) for y in key_points[1::2]]
    center_x = int(sum(separate_x) / len(separate_x))
    center_y = int(sum(separate_y) / len(separate_y))
    return center_x,center_y


def compute_bbox(key_points):
    separate_x = [int(x) for x in key_points[::2]]
    separate_y = [int(y) for y in key_points[1::2]]
    min_x = min(separate_x)
    max_x = max(separate_x)
    min_y = min(separate_y)
    max_y = max(separate_y)
    return [(min_x, min_y), (max_x, max_y)]


INDEX = 0


def draw_bboxes(image_path, b_boxes, offset_x=20, offset_y=80):
    global INDEX
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    for i in range(len(b_boxes)):
        image = np.array(image)
        cv2.rectangle(image, (b_boxes[i][0][0] - offset_x, b_boxes[i][0][1] - offset_y),
                      (b_boxes[i][1][0] + offset_x, b_boxes[i][1][1] + offset_y), (255, 255, 255), thickness=2)
        # copy_image = image.copy()
        # cropped_image = copy_image[(b_boxes[i][0][0] - offset_x):(b_boxes[i][0][1] - offset_y),(b_boxes[i][1][0] + offset_x):(b_boxes[i][1][1] + offset_y)]
        # image_path_crop = POSE_DIR + str(INDEX) + ".jpg"
        # INDEX += 1
        # cv2.imwrite(image_path_crop, cropped_image)

    cv2.imwrite(image_path, image)


def read_csv(image_path, csv_path, camera, pixels):
    header = []
    rows = []
    centers_list = []
    b_boxes = []
    with open(csv_path, 'r') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)
        for row in csv_reader:
            rows.append(row)
            centers_list.append(compute_center(row))
            b_boxes.append(compute_bbox(row))
    draw_bboxes(image_path, b_boxes)
    centers = np.array(centers_list)
    pixels[camera] = centers
    return pixels


def find_id_position(point, fused_points):
    human_width = 0.57
    for i, pt in enumerate(fused_points):
        if np.linalg.norm(point - pt) <= human_width:
            return i
    return None


def assign_ids(cameras, pixels, fused_pt, image_paths, ids, frame_id):
    for i, cam in enumerate(cameras):
        pxls = pixels[cam]
        for px in pxls:
            point = reproject_pixel_in_3D(cam, px, frame_id)
            if point is not None:
                id_pos = find_id_position(point, fused_pt)
                if id_pos is not None:
                    image = cv2.imread(image_paths[i], cv2.IMREAD_UNCHANGED)
                    image = cv2.putText(image, f"id:{ids[id_pos]}", (px[0], px[1]),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                                       2, lineType=cv2.LINE_AA)
                    cv2.imwrite(image_paths[i], image)


INPUT_DIR = "/media/ana/Multimedia/UbuntuChestii/bodytracking_all/"
# INPUT_DIR = "/home/anaml/bodytracking/"
OUTPUT_DIR = "/home/ana/PycharmProjects/DEKR/output/"
# OUTPUT_DIR = "/home/anaml/DEKR/output/"
CAMERAS = ["cn01","cn02", "cn03", "cn04", "cn05", "cn06"]


def create(frame_id, dir=INPUT_DIR, total_padding=10, ending="_color.jpg"):
    image_paths = []
    frame = frame_id.zfill(total_padding)
    frame = frame+ ending
    for cam in CAMERAS:
        image_paths.append(os.path.join(dir, cam, frame))
    return image_paths


def create_csv(name="pose-data-", dir=OUTPUT_DIR, ending=".csv"):
    csv_paths = []
    for cam in CAMERAS:
        file = name + cam + ending
        csv_paths.append(os.path.join(dir, file))
    return csv_paths


def re_identify(frame_id):
    image_paths_input = create(frame_id)
    image_paths_input = zip(image_paths_input, CAMERAS)
    image_paths_output = []
    for image_path, camera in image_paths_input:
        image_paths_output.append(main(image_path, camera))

    pixels = {}
    csv_paths = create_csv()

    image_paths_output_zipped = zip(image_paths_output, csv_paths, CAMERAS)

    for image_path, csv_path, camera in image_paths_output_zipped:
        pixels = read_csv(image_path, csv_path, camera, pixels)

    world_pts = np.array(construct_world_points(pixels, frame_id))
    # visualize(world_pts, frame_id, multiplePoints=True)
    fused_pts, ids = fuse_world_points(world_pts)
    # visualize(fused_pts, frame_id, multiplePoints=True)

    assign_ids(CAMERAS, pixels, fused_pts, image_paths_output, ids, frame_id)


if __name__ == '__main__':
    frame_ids = ["1745","1755","1760", "1765","1770", "1775", "1790"]

    for frame_id in frame_ids:
        re_identify(frame_id)

    # extract_frames()

