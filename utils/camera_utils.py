#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import numpy as np
from PIL import Image

from scene.cameras import Camera
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal, focal2fov

WARNED = False

def read_image_and_update_cam_info(cam_info):

    # Camera intrinsics fx, fy, cx, and cy need to be scaled if original image is rescaled (intrinsics are recorded for original image size)
    # Distortion coefficients do not change
    fx = cam_info.fx
    fy = cam_info.fy
    cx = cam_info.cx
    cy = cam_info.cy
    crop_edge = cam_info.crop_edge
    depth_scale = cam_info.depth_scale


    image_color = Image.open(cam_info.image_path)
    image_depth = (
        np.asarray(Image.open(cam_info.depth_path), dtype=np.float32) / depth_scale
    )
    image_color = np.asarray(
        image_color.resize((image_depth.shape[1], image_depth.shape[0]))
    )
    if crop_edge > 0:
        image_color = image_color[
            crop_edge:-crop_edge,
            crop_edge:-crop_edge,
            :,
        ]
        image_depth = image_depth[
            crop_edge:-crop_edge,
            crop_edge:-crop_edge,
        ]
        cx -= crop_edge
        cy -= crop_edge

    # Image height and width is set based on image - overrides the random intialization in the data_reader.py file
    height, width = image_color.shape[:2]
    # print("image size:", height, width)


    FovX = focal2fov(fx, width)
    FovY = focal2fov(fy, height)

    cam_info.image=Image.fromarray(image_color)
    cam_info.depth=Image.fromarray(image_depth)
    cam_info.FovX = FovX
    cam_info.FovY = FovY

    return

def loadCam(args, id, cam_info, resolution_scale):

    read_image_and_update_cam_info(cam_info)

    orig_w, orig_h = cam_info.image.size
    preload = args.preload
    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w / (resolution_scale * args.resolution)), round(
            orig_h / (resolution_scale * args.resolution)
        )
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print(
                        "[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1"
                    )
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)
    resized_image_depth = PILtoTorch(cam_info.depth, resolution, Image.NEAREST)
    gt_image = resized_image_rgb[:3, ...]
    gt_depth = resized_image_depth
    loaded_mask = None
    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(
        colmap_id=cam_info.uid,
        R=cam_info.R,
        T=cam_info.T,
        FoVx=cam_info.FovX,
        FoVy=cam_info.FovY,
        image=gt_image,
        depth=gt_depth,
        gt_alpha_mask=loaded_mask,
        image_name=cam_info.image_name,
        uid=id,
        data_device=args.data_device,
        pose_gt=cam_info.pose_gt,
        cx=cam_info.cx / resolution_scale,
        cy=cam_info.cy / resolution_scale,
        timestamp=cam_info.timestamp,
        preload=preload,
        depth_scale=cam_info.depth_scale,
    )


def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list


def camera_to_JSON(id, camera: Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        "id": id,
        "img_name": camera.image_name,
        "width": camera.width,
        "height": camera.height,
        "position": pos.tolist(),
        "rotation": serializable_array_2d,
        "fy": fov2focal(camera.FovY, camera.height),
        "fx": fov2focal(camera.FovX, camera.width),
    }
    return camera_entry
