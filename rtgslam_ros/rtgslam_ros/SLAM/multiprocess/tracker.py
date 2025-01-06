import copy
import matplotlib.pyplot as plt
from rtgslam_ros.SLAM.gaussian_pointcloud import *

import torch.multiprocessing as mp
from rtgslam_ros.SLAM.render import Renderer
from collections import defaultdict
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

from rtgslam_ros.SLAM.icp import IcpTracker
from threading import Thread
from rtgslam_ros.utils.camera_utils import loadCam

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import threading

from rtgslam_interfaces.msg import F2B, B2F, F2G, Camera, Gaussian
from rtgslam_ros.utils.ros_utils import (
    convert_ros_array_message_to_tensor, 
    convert_ros_multi_array_message_to_tensor, 
    convert_tensor_to_ros_message, 
    convert_numpy_array_to_ros_message, 
    convert_ros_multi_array_message_to_numpy, 
)
# from rtgslam_ros.src.gui.gui_utils import (
#     ParamsGUI,
#     GaussianPacket,
#     Packet_vis2main,
#     create_frustum,
#     cv_gl,
#     get_latest_queue,
# )
# from rtgslam_ros.src.utils.multiprocessing_utils import clone_obj


import os
from argparse import ArgumentParser
from rtgslam_ros.utils.config_utils import read_config
from rtgslam_ros.arguments import DatasetParams, MapParams, OptimizationParams
from rtgslam_ros.scene import Dataset
from rtgslam_ros.SLAM.multiprocess.mapper import MappingProcess
from rtgslam_ros.utils.general_utils import safe_state

def convert_poses(trajs):
    poses = []
    stamps = []
    for traj in trajs:
        stamp, r00, r01, r02, t0, r10, r11, r12, t1, r20, r21, r22, t2 = traj
        pose_ = np.eye(4)
        pose_[:3, :3] = np.array([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]])
        pose_[:3, 3] = np.array([t0, t1, t2])
        poses.append(pose_)
        stamps.append(stamp)
    return poses, stamps

class Tracker(Node):
    def __init__(self, args):
        super().__init__('tracker_frontend_node')

        self.use_gt_pose = args.use_gt_pose
        self.mode = args.mode
        self.K = None

        self.min_depth = args.min_depth
        self.max_depth = args.max_depth
        self.depth_filter = args.depth_filter
        self.verbose = args.verbose

        self.icp_tracker = IcpTracker(args)

        self.status = defaultdict(bool)
        self.pose_gt = []
        self.pose_es = []
        self.timestampes = []
        self.finish = False # mp.Condition() # Need to find another way of signalling the end - TBD

        self.icp_success_count = 0

        self.use_orb_backend = args.use_orb_backend
        self.orb_vocab_path = args.orb_vocab_path
        self.orb_settings_path = args.orb_settings_path
        self.orb_backend = None
        self.orb_useicp = args.orb_useicp

        self.invalid_confidence_thresh = args.invalid_confidence_thresh

        if self.mode == "single process":
            self.initialize_orb()

    def get_new_poses_byid(self, frame_ids):
        if self.use_orb_backend and not self.use_gt_pose:
            new_poses = convert_poses(self.orb_backend.get_trajectory_points())
            frame_poses = [new_poses[frame_id] for frame_id in frame_ids]
        else:
            frame_poses = [self.pose_es[frame_id] for frame_id in frame_ids]
        return frame_poses

    def get_new_poses(self):
        if self.use_orb_backend and not self.use_gt_pose:
            new_poses, _ = convert_poses(self.orb_backend.get_trajectory_points())
        else:
            new_poses = None
        return new_poses

    def save_invalid_traing(self, path):
        if np.linalg.norm(self.pose_es[-1][:3, 3] - self.pose_gt[-1][:3, 3]) > 0.15:
            if self.track_mode == "icp":
                frame_id = len(self.pose_es)
                torch.save(
                    self.icp_tracker.vertex_pyramid_t1,
                    os.path.join(path, "vertex_pyramid_t1_{}.pt".format(frame_id)),
                )
                torch.save(
                    self.icp_tracker.vertex_pyramid_t0,
                    os.path.join(path, "vertex_pyramid_t0_{}.pt".format(frame_id)),
                )
                torch.save(
                    self.icp_tracker.normal_pyramid_t1,
                    os.path.join(path, "normal_pyramid_t1_{}.pt".format(frame_id)),
                )
                torch.save(
                    self.icp_tracker.normal_pyramid_t0,
                    os.path.join(path, "normal_pyramid_t0_{}.pt".format(frame_id)),
                )

    def map_preprocess(self, frame, frame_id):
        depth_map, color_map = (
            frame.original_depth.permute(1, 2, 0) * 255,
            frame.original_image.permute(1, 2, 0),
        )  # [H, W, C], the image is scaled by 255 in function "PILtoTorch"
        depth_map_orb = (
            frame.original_depth.permute(1, 2, 0).cpu().numpy()
            * 255
            * frame.depth_scale
        ).astype(np.uint16)
        intrinsic = frame.get_intrinsic
        # depth filter
        if self.depth_filter:
            depth_map_filter = bilateralFilter_torch(depth_map, 5, 2, 2)
        else:
            depth_map_filter = depth_map

        valid_range_mask = (depth_map_filter > self.min_depth) & (depth_map_filter < self.max_depth)
        depth_map_filter[~valid_range_mask] = 0.0
        # update depth map
        frame.original_depth = depth_map_filter.permute(2, 0, 1) / 255.0
        # compute geometry info
        vertex_map_c = compute_vertex_map(depth_map_filter, intrinsic)
        normal_map_c = compute_normal_map(vertex_map_c)
        confidence_map = compute_confidence_map(normal_map_c, intrinsic)

        # confidence_threshold tum: 0.5, others: 0.2
        invalid_confidence_mask = ((normal_map_c == 0).all(dim=-1)) | (
            confidence_map < self.invalid_confidence_thresh
        )[..., 0]

        depth_map_filter[invalid_confidence_mask] = 0
        normal_map_c[invalid_confidence_mask] = 0
        vertex_map_c[invalid_confidence_mask] = 0
        confidence_map[invalid_confidence_mask] = 0

        color_map_orb = (
            (frame.original_image * 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        )

        self.update_curr_status(
            frame,
            frame_id,
            depth_map,
            depth_map_filter,
            vertex_map_c,
            normal_map_c,
            color_map,
            color_map_orb,
            depth_map_orb,
            intrinsic,
        )

        frame_map = {}
        frame_map["depth_map"] = depth_map_filter
        frame_map["color_map"] = color_map
        frame_map["normal_map_c"] = normal_map_c
        frame_map["vertex_map_c"] = vertex_map_c
        frame_map["confidence_map"] = confidence_map
        frame_map["invalid_confidence_mask"] = invalid_confidence_mask
        frame_map["time"] = frame_id
        frame_map["finish"] = False

        return frame_map

    def update_curr_status(
        self,
        frame,
        frame_id,
        depth_t1,
        depth_t1_filter,
        vertex_t1,
        normal_t1,
        color_t1,
        color_orb,
        depth_orb,
        K,
    ):
        if self.K is None:
            self.K = K
        self.curr_frame = {
            "K": frame.get_intrinsic,
            "normal_map": normal_t1,
            "depth_map": depth_t1,
            "depth_map_filter": depth_t1_filter,
            "vertex_map": vertex_t1,
            "frame_id": frame_id,
            "pose_gt": frame.get_c2w.cpu().numpy(), # 1
            "color_map": color_t1,
            "timestamp": frame.timestamp, # 1
            "color_map_orb": color_orb, # 1
            "depth_map_orb": depth_orb, # 1
        }
        self.icp_tracker.update_curr_status(depth_t1_filter, self.K)
        
    def update_last_status_v2(
        self, frame, render_depth, frame_depth, render_normal, frame_normal
    ):
        intrinsic = frame.get_intrinsic
        normal_mask = (
            1 - F.cosine_similarity(render_normal, frame_normal, dim=-1)
        ) < self.icp_sample_normal_threshold
        depth_filling_mask = (
            (
                torch.abs(render_depth - frame_depth)
                > self.icp_sample_distance_threshold
            )[..., 0]
            | (render_depth == 0)[..., 0]
            | (normal_mask)
        ) & (frame_depth > 0)[..., 0]

        render_depth[depth_filling_mask] = frame_depth[depth_filling_mask]
        render_depth[(frame_depth == 0)[..., 0]] = 0
        
        self.last_model_vertex = compute_vertex_map(render_depth, intrinsic)
        self.last_model_normal = compute_normal_map(self.last_model_vertex)

    def update_last_status(
        self,
        frame,
        render_depth,
        frame_depth,
        render_normal,
        frame_normal,
    ):
        self.icp_tracker.update_last_status(
            frame, render_depth, frame_depth, render_normal, frame_normal
        )

    def refine_icp_pose(self, pose_t1_t0, tracking_success):
        if tracking_success and self.orb_useicp:
            print("success")
            self.orb_backend.track_with_icp_pose(
                self.curr_frame["color_map_orb"],
                self.curr_frame["depth_map_orb"],
                pose_t1_t0.astype(np.float32),
                self.curr_frame["timestamp"],
            )
            time.sleep(0.005)
        else:
            self.orb_backend.track_with_orb_feature(
                self.curr_frame["color_map_orb"],
                self.curr_frame["depth_map_orb"],
                self.curr_frame["timestamp"],
            )
            time.sleep(0.005)
        traj_history = self.orb_backend.get_trajectory_points()
        pose_es_t1, _ = convert_poses(traj_history[-2:])
        return pose_es_t1[-1]

    def initialize_orb(self):
        if not self.use_gt_pose and self.use_orb_backend and self.orb_backend is None:
            import orbslam2
            print("init orb backend")
            self.orb_backend = orbslam2.System(
                self.orb_vocab_path, self.orb_settings_path, orbslam2.Sensor.RGBD
            )
            self.orb_backend.set_use_viewer(False)
            self.orb_backend.initialize(self.orb_useicp)

    def initialize_tracker(self):
        if self.use_orb_backend:
            self.orb_backend.process_image_rgbd(
                self.curr_frame["color_map_orb"],
                self.curr_frame["depth_map_orb"],
                self.curr_frame["timestamp"],
            )
        self.status["initialized"] = True

    def tracking(self, frame, frame_map):
        self.pose_gt.append(self.curr_frame["pose_gt"])
        self.timestampes.append(self.curr_frame["timestamp"])
        p2loss = 0
        tracking_success = True
        if self.use_gt_pose:
            pose_t1_w = self.pose_gt[-1]
        else:
            # initialize
            if not self.status["initialized"]:
                self.initialize_tracker()
                pose_t1_w = np.eye(4)
            else:
                pose_t1_t0, tracking_success = self.icp_tracker.predict_pose(self.curr_frame)
                if self.use_orb_backend:
                    pose_t1_w = self.refine_icp_pose(pose_t1_t0, tracking_success)
                else:
                    pose_t1_w = self.pose_es[-1] @ pose_t1_t0

        self.icp_tracker.move_last_status()
        self.pose_es.append(pose_t1_w)

        frame.updatePose(pose_t1_w)
        frame_map["vertex_map_w"] = transform_map(
            frame_map["vertex_map_c"], frame.get_c2w
        )
        frame_map["normal_map_w"] = transform_map(
            frame_map["normal_map_c"], get_rot(frame.get_c2w)
        )

        return tracking_success

    def eval_total_ate(self, pose_es, pose_gt):
        ates = []
        for i in tqdm(range(1, len(pose_gt) + 1)):
            ates.append(self.eval_ate(pose_es, pose_gt, i))
        ates = np.array(ates)
        return ates

    def save_ate_fig(self, ates, save_path, save_name):
        plt.plot(range(len(ates)), ates)
        plt.ylim(0, max(ates) + 0.1)
        plt.title("ate:{}".format(ates[-1]))
        plt.savefig(os.path.join(save_path, "{}.png".format(save_name)))
    

    def save_keyframe_traj(self, save_file):
        if self.use_orb_backend:
            poses, stamps = convert_poses(self.orb_backend.get_keyframe_points())
            with open(save_file, "w") as f:
                for pose_id, pose_es_ in enumerate(poses):
                    t = pose_es_[:3, 3]
                    q = R.from_matrix(pose_es_[:3, :3])
                    f.write(str(stamps[pose_id]) + " ")
                    for i in t.tolist():
                        f.write(str(i) + " ")
                    for i in q.as_quat().tolist():
                        f.write(str(i) + " ")
                    f.write("\n")

    def save_traj_tum(self, save_file):
        poses, stamps = convert_poses(self.orb_backend.get_trajectory_points())
        with open(save_file, "w") as f:
            for pose_id, pose_es_ in enumerate(self.pose_es):
                t = pose_es_[:3, 3]
                q = R.from_matrix(pose_es_[:3, :3])
                f.write(str(stamps[pose_id]) + " ")
                for i in t.tolist():
                    f.write(str(i) + " ")
                for i in q.as_quat().tolist():
                    f.write(str(i) + " ")
                f.write("\n")

    def save_orb_traj_tum(self, save_file):
        if self.use_orb_backend:
            poses, stamps = convert_poses(self.orb_backend.get_trajectory_points())
            with open(save_file, "w") as f:
                for pose_id, pose_es_ in enumerate(poses):
                    t = pose_es_[:3, 3]
                    q = R.from_matrix(pose_es_[:3, :3])
                    f.write(str(stamps[pose_id]) + " ")
                    for i in t.tolist():
                        f.write(str(i) + " ")
                    for i in q.as_quat().tolist():
                        f.write(str(i) + " ")
                    f.write("\n")

    def save_traj(self, save_path):
        save_traj_path = os.path.join(save_path, "save_traj")
        if not self.use_gt_pose and self.use_orb_backend:
            traj_history = self.orb_backend.get_trajectory_points()
            self.pose_es, _ = convert_poses(traj_history)
        pose_es = np.stack(self.pose_es, axis=0)
        pose_gt = np.stack(self.pose_gt, axis=0)
        ates_ba = self.eval_total_ate(pose_es, pose_gt)
        print("ate: ", ates_ba[-1])
        np.save(os.path.join(save_traj_path, "pose_gt.npy"), pose_gt)
        np.save(os.path.join(save_traj_path, "pose_es.npy"), pose_es)
        self.save_ate_fig(ates_ba, save_traj_path, "ate")

        plt.figure()
        plt.plot(pose_es[:, 0, 3], pose_es[:, 1, 3])
        plt.plot(pose_gt[:, 0, 3], pose_gt[:, 1, 3])
        plt.legend(["es", "gt"])
        plt.savefig(os.path.join(save_traj_path, "traj_xy.jpg"))
        
        if self.use_orb_backend:
            self.orb_backend.shutdown()
        
    def eval_ate(self, pose_es, pose_gt, frame_id=-1):
        pose_es = np.stack(pose_es, axis=0)[:frame_id, :3, 3]
        pose_gt = np.stack(pose_gt, axis=0)[:frame_id, :3, 3]
        ate = eval_ate(pose_gt, pose_es)
        return ate


class TrackingProcess(Tracker):
    def __init__(self, map_params, optimization_params, dataset, args):
        args.icp_use_model_depth = False
        super().__init__(args)

        self.args = args
        # online scanner
        self.use_online_scanner = args.use_online_scanner
        self.scanner_finish = False
        self.device = "cuda"

        self.map_params = map_params

        # sync mode
        self.sync_tracker2mapper_method = self.map_params.sync_tracker2mapper_method
        self.sync_tracker2mapper_frames = self.map_params.sync_tracker2mapper_frames
        self.rx_backend_msg = False

        self.mapper_running = True


        self.frame = None
        self.frame_id = None
        self.pause = False
        self.requested_init = False
        self.reset = True


        self.dataset = dataset

        self.dataset_cameras = self.dataset.scene_info.train_cameras
        #self.map_process = MappingProcess(args, optimization_params, self) - Dont need this anymore
        self._end = False
        self.max_fps = args.tracker_max_fps
        self.frame_time = 1.0 / self.max_fps
        self.frame_id = 0
        self.last_mapper_frame_id = 0

        self.last_frame = None
        self.last_global_params = None

        self.track_renderer = Renderer(args)
        self.save_path = args.save_path

        self.queue_size = 100
        self.msg_counter = 0
        self.f2b_publisher = self.create_publisher(F2B,'/Front2Back',self.queue_size)
        self.f2g_publisher = self.create_publisher(F2G,'/Front2GUI',self.queue_size)
        self.b2f_subscriber = self.create_subscription(B2F, '/Back2Front', self.b2f_listener_callback, self.queue_size)
        self.b2f_subscriber  # prevent unused variable warning

    def map_preprocess_mp(self, frame, frame_id):
        self.map_input = super().map_preprocess(frame, frame_id)

    def getNextFrame(self):
        frame_info = self.dataset_cameras[self.frame_id]
        frame = loadCam(self.args, self.frame_id, frame_info, 1)
        print("get frame: {}".format(self.frame_id))
        self.frame_id += 1
        return frame

    def convert_to_f2b_ros_msg(self):

        f2b_msg = F2B()

        f2b_msg.msg = f'Hello world {self.msg_counter}'

        # Sending the frame
        f2b_msg.frame.uid = self.map_input["frame"].uid
        f2b_msg.frame.rot = convert_numpy_array_to_ros_message(self.map_input["frame"].R)
        f2b_msg.frame.trans = self.map_input["frame"].T.tolist()
        f2b_msg.frame.fovx = self.map_input["frame"].FoVx
        f2b_msg.frame.fovy = self.map_input["frame"].FoVy
        f2b_msg.frame.timestamp = self.map_input["frame"].timestamp
        f2b_msg.frame.depth_scale = self.map_input["frame"].depth_scale
        f2b_msg.frame.original_image = convert_tensor_to_ros_message(self.map_input["frame"].original_image)
        f2b_msg.frame.image_width = self.map_input["frame"].image_width
        f2b_msg.frame.image_height = self.map_input["frame"].image_height
        f2b_msg.frame.original_depth = convert_tensor_to_ros_message(self.map_input["frame"].original_depth)
        f2b_msg.frame.cx = self.map_input["frame"].cx
        f2b_msg.frame.cy = self.map_input["frame"].cy

        f2b_msg.time = self.map_input["time"]
        f2b_msg.color_map = convert_tensor_to_ros_message(self.map_input["color_map"])
        f2b_msg.depth_map = convert_tensor_to_ros_message(self.map_input["depth_map"])
        f2b_msg.normal_map_c = convert_tensor_to_ros_message(self.map_input["normal_map_c"])
        f2b_msg.normal_map_w = convert_tensor_to_ros_message(self.map_input["normal_map_w"])
        f2b_msg.vertex_map_c = convert_tensor_to_ros_message(self.map_input["vertex_map_c"])
        f2b_msg.vertex_map_w = convert_tensor_to_ros_message(self.map_input["vertex_map_w"])
        f2b_msg.confidence_map = convert_tensor_to_ros_message(self.map_input["confidence_map"])
        #f2b_msg.invalid_confidence_mask = convert_tensor_to_ros_message(self.map_input["invalid_confidence_mask"]) how to handle torch.BoolTensor
        f2b_msg.finish = self.map_input["finish"]
        if self.map_input["poses_new"] is not None:
            f2b_msg.poses_new = convert_tensor_to_ros_message(self.map_input["poses_new"])

        return f2b_msg

    def publish_to_backend(self, msg="default"):
        # Implement sending info to mapper via ROS topics
        print("tracker send frame {} to mapper".format(self.map_input["time"]))
        f2b_msg = self.convert_to_f2b_ros_msg()
        f2b_msg.msg = msg
        self.get_logger().info(f'{f2b_msg.msg}: Publishing to Backend Node: {self.msg_counter}')
        self.f2b_publisher.publish(f2b_msg)
        self.msg_counter += 1


# Requirements:

# 1) We want the environment to be mapped rapidly (as rapid as possible) at least the image acquisition should be rapid.
# - Robot exploration speed will be limited by below factors
# a) Speed of the tracker and mapper determine how fast robots explore the space (need to profile these algorithms for running time)
# b) Blurred images are useless (??) - Blur speed of the cameras needs to be measured

# 2) Ideally we want to automate this process (Active SLAM) - for now we can assume it is done manually
# (i) If manually done, you have to signal to the teleoperator to pause (when tracker and mapper need to catchup with each other, or tracking failure events)
# (a) Live reconstruction may be useful to the teleoperator - spend more time in poorly reconstructed areas, problematic areas, things that require zoom-in, holes, etc.

# (ii) For automated exploration
# (a) How is the automated exploration planned?
# - Sweep out the outer boundary of an unknown area and then explore its interiors in a raster form??
# - Explore any blind zones that remained one at a time??
# (b) Since, one cant rely on the map for movement and collision avoidance, one may need to use LIDARS or point clouds and safety distance buffers?

# Options to setup coordination between tracker and mapper are:

# A) Continue to track the frames without relying on the Gaussian Map at all (such as when Tracker is ORBSLAM)
# - you can run the map optimization at your own pace in that case and focus on accuracy

# Key questions that need to be answered if we were to use this approach

# A1) Does incorrect poses lead to poor maps? (MonoGS paper i think talks of good convergence as long as you are in the vicinity)
# A2) Can you get away by running the optimization for more number of iterations?
# A3) What are the limitations of such an approach?

# b) Continue to track the frames without relying on the Gaussian Map for sometime
# - pause when tracked frame id and mapped frame id are not in sync (for this robot motion has to stop)

# c) Tracker in sync with Gaussian mapper for every frame (robot motion for exploration will need to be very slow)

# Some key questions: 
# 1) Compare and contrast how fast are the various trackers out there (ORBSLAM, DGO, Gaussian Opt Tracker, ICP-GS).
# 2) Does the accuracy of the tracking improve with Gaussian model based tracking?
# 3) Effect of blurred images on the map?
# - ultimately robot speed of exploration is limited by blur_speed, tracker_speed and 
# 4) Does incorrect poses lead to poor maps? Can the



    def acquire_frame_and_track(self, msg="default"):

        self.frame = self.getNextFrame()
        if self.frame is None:
            print("Need help!!! frame is returning None....")
        self.frame_id = self.frame.uid
        print("current tracker frame = %d" % self.time)
        # update current map
        move_to_gpu(self.frame)

        self.map_preprocess_mp(self.frame, self.frame_id)
        self.tracking(self.frame, self.map_input)
        self.map_input["frame"] = copy.deepcopy(self.frame)
        self.map_input["frame"] = self.frame
        self.map_input["poses_new"] = self.get_new_poses() # Depends on ORBSLAM backend for tracker

        self.publish_to_backend(msg)
        self.update_last_mapper_render(self.frame)

        # # Reimplement - Publisher T2G and corresponding messages
        # self.update_viewer(self.frame)

        move_to_cpu(self.frame)

        self.time += 1

    def run(self):
        self.time = 0
        self.initialize_orb()

        while not self.finish_():

            if self.pause:
                continue

            if self.requested_init:
                # self.requested_init is made False in the listener after hearing from mapper
                print("Waiting for reply from mapper to init message....")
                continue

            if self.reset:

                # All code to initialize and reset system to be put here

                # Acquire First Frame
                self.acquire_frame_and_track("init")
                self.requested_init = True
                self.reset = False

            # Reach here if not paused, reset is done, requested_init is acknowledged by mapper
            # At this point you have a gaussian map and the next frame
            
            # Handle synchronization here
            # For now it is strictly back and forth - comes at the cost of real-time operation and not even realistic as robot needs to pause exploration

            # Handle keyframe management here - for now nothing to be done as every frame is a keyframe

        # Signal to mapper to finish and save trajectory - Needs to be reimplemented (can have a boolean variable signalling this when sending each frame )
        self.map_input["finish"] = True
        self.publish_to_backend("finish")
        #self.save_traj(self.save_path)
        print("track finish")


    def convert_from_b2f_ros_msg(self, b2f_msg):
        last_frame_id = b2f_msg.frame_id

        frame_info = self.dataset_cameras[last_frame_id]
        last_frame = loadCam(self.args, last_frame_id, frame_info, 1)

        last_frame.R = convert_ros_multi_array_message_to_tensor(b2f_msg.frame.rot, self.device)
        last_frame.T = convert_ros_multi_array_message_to_tensor(b2f_msg.frame.trans, self.device)


        submap_gaussians = GaussianPointCloud(self.args)
        # submap_gaussians.training_setup(self.opt)

        submap_gaussians._xyz = convert_ros_multi_array_message_to_tensor(b2f_msg.xyz, self.device)
        submap_gaussians._opacity = convert_ros_multi_array_message_to_tensor(b2f_msg.opacity, self.device)
        submap_gaussians._scaling = convert_ros_multi_array_message_to_tensor(b2f_msg.scales, self.device)
        submap_gaussians._rotation = convert_ros_multi_array_message_to_tensor(b2f_msg.rotation, self.device)

        submap_gaussians._shs = convert_ros_multi_array_message_to_tensor(b2f_msg.shs, self.device) # How to convert shs to features (both dc and rest??)
        #submap_gaussians._features_dc = convert_ros_multi_array_message_to_tensor(b2f_msg.gaussian.features_dc, self.device)
        #submap_gaussians._features_rest = convert_ros_multi_array_message_to_tensor(b2f_msg.gaussian.features_rest, self.device)
        

        #submap_gaussians.radius = convert_ros_array_message_to_tensor(b2f_msg.radius, self.device)
        submap_gaussians.normal = convert_ros_array_message_to_tensor(b2f_msg.normal, self.device)
        submap_gaussians.confidence = convert_ros_array_message_to_tensor(b2f_msg.confidence, self.device)

        return last_frame_id, last_frame, submap_gaussians

    def b2f_listener_callback(self, b2f_msg):
        self.get_logger().info(f'Rx from Backend Node {b2f_msg.msg}')
        self.last_mapper_frame_id, self.last_frame, self.last_global_params = self.convert_from_b2f_ros_msg(b2f_msg)

        if b2f_msg.msg == "init":
            # Received acknowledgement for init message
            self.requested_init = False
        if b2f_msg.msg == "sync":
            pass

        # Acquire and track next keyframe
        self.acquire_frame_and_track("keyframe")
        



    def finish_(self):
        if self.use_online_scanner:
            return self.scanner_finish
        else:
            return self.frame_id >= len(self.dataset_cameras)

    def update_viewer(self, frame):
        #Implement info to be passed to GUI here
        pass

    def unpack_map_to_tracker(self):
        pass
        # self._mapper2tracker_call.acquire()
        # while not self._mapper2tracker_map_queue.empty():
        #     map_info = self._mapper2tracker_map_queue.get()
        #     self.last_frame = map_info["frame"]
        #     self.last_global_params = map_info["global_params"]
        #     self.last_mapper_frame_id = map_info["frame_id"]
        #     print("tracker unpack map {}".format(self.last_mapper_frame_id))
        # self._mapper2tracker_call.notify()
        # self._mapper2tracker_call.release()

    def update_last_mapper_render(self, frame):
        pose_t0_w = frame.get_c2w.cpu().numpy()
        if self.last_frame is not None:
            pose_w_t0 = np.linalg.inv(pose_t0_w)
            self.last_frame.update(pose_w_t0[:3, :3].transpose(), pose_w_t0[:3, 3])
            render_output = self.track_renderer.render(
                self.last_frame,
                self.last_global_params,
                None
            )
            self.update_last_status(
                self.last_frame,
                render_output["depth"].permute(1, 2, 0),
                self.map_input["depth_map"],
                render_output["normal"].permute(1, 2, 0),
                self.map_input["normal_map_w"],
            )

    # def stop(self):
    #     with self.finish:
    #         self.finish.notify()

def spin_thread(node):
    # Spin the node continuously in a separate thread
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)

def main():
    parser = ArgumentParser(description="Training script parameters")
    args = parser.parse_args()
    config_path = "/root/code/rtgslam_ros_ws/src/rtgslam_ros/rtgslam_ros/configs/tum/fr1_desk.yaml"
    args = read_config(config_path)
    # set visible devices
    device_list = args.device_list
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(device) for device in device_list)

    safe_state(args.quiet)
    
    optimization_params = OptimizationParams(parser)
    optimization_params = optimization_params.extract(args)

    dataset_params = DatasetParams(parser, sentinel=True)
    dataset_params = dataset_params.extract(args)
    
    map_params = MapParams(parser)
    map_params = map_params.extract(args)

    # Initialize dataset
    dataset = Dataset(
        dataset_params,
        shuffle=False,
        resolution_scales=dataset_params.resolution_scales,
    )

    rclpy.init()
    tracker_node = TrackingProcess(map_params, optimization_params, dataset, args)
    try:
        # Start the spin thread for continuously handling callbacks
        spin_thread_instance = threading.Thread(target=spin_thread, args=(tracker_node,))
        spin_thread_instance.start()

        # Run the main logic (this will execute in parallel with message handling)
        tracker_node.run()
        
    finally:
        tracker_node.destroy_node()
        rclpy.shutdown()
        spin_thread_instance.join()  # Wait for the spin thread to finish