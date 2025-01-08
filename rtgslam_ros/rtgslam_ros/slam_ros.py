import os
from argparse import ArgumentParser

from rtgslam_ros.utils.config_utils import read_config

parser = ArgumentParser(description="Training script parameters")
parser.add_argument("--config", type=str, default="/root/code/rtgslam_ros_ws/src/rtgslam_ros/rtgslam_ros/configs/tum/fr1_desk.yaml")
args = parser.parse_args()
config_path = args.config
args = read_config(config_path)
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(device) for device in args.device_list)

import torch
import json
from rtgslam_ros.utils.camera_utils import loadCam
from rtgslam_ros.arguments import DatasetParams, MapParams, OptimizationParams
from rtgslam_ros.scene import Dataset
from rtgslam_ros.SLAM.multiprocess.mapper import Mapping
from rtgslam_ros.SLAM.multiprocess.tracker import Tracker
from rtgslam_ros.SLAM.utils import *
# from rtgslam_ros.SLAM.eval import eval_frame
from rtgslam_ros.utils.general_utils import safe_state
from rtgslam_ros.utils.monitor import Recorder


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
from rtgslam_ros.gui.gui_utils import GaussianPacket, clone_obj
from rtgslam_ros.SLAM.gaussian_pointcloud import *


torch.set_printoptions(4, sci_mode=False)


class SLAM_ROS(Node):
    def __init__(self, args):
        super().__init__('slam_ros_node')
        self.args = args

        self.queue_size = 100
        self.msg_counter = 0
        self.f2g_publisher = self.create_publisher(F2G,'/Front2GUI',self.queue_size)


    def get_gaussian_packet(self, current_frame, last_global_params):
        gaussian_model = GaussianPointCloud(self.args)
        gaussian_model._xyz = last_global_params["xyz"].clone().detach()
        gaussian_model._opacity = last_global_params["opacity"].clone().detach()
        gaussian_model._scaling = last_global_params["scales"].clone().detach()
        gaussian_model._rotation = last_global_params["rotations"].clone().detach()
        
        shs = last_global_params["shs"].clone().detach()
        gaussian_model._features_dc = shs[:, 0:1, :]
        gaussian_model._features_rest = shs[:, 1:, :]

        gaussian_model._normal = last_global_params["normal"].clone().detach()
        gaussian_model._confidence = last_global_params["confidence"].clone().detach()

        gaussian_packet = GaussianPacket(
            gaussians=clone_obj(gaussian_model),
            current_frame=current_frame,
            gtcolor=current_frame.original_image,
            gtdepth=current_frame.original_depth
        )
        
        return gaussian_packet

    def publish_message_to_gui(self, gaussian_packet):
        f2g_msg = self.convert_to_f2g_ros_msg(gaussian_packet)
        f2g_msg.msg = f'Hello world {self.msg_counter}'
        self.get_logger().info(f'Publishing to GUI Node: {self.msg_counter}')

        self.f2g_publisher.publish(f2g_msg)
        self.msg_counter += 1


    def convert_to_f2g_ros_msg(self, gaussian_packet):
        
        f2g_msg = F2G()

        f2g_msg.msg = "Sending 3D Gaussians"
        f2g_msg.has_gaussians = gaussian_packet.has_gaussians
        #f2g_msg.submap_id = gaussian_packet.submap_id

        if gaussian_packet.has_gaussians:
            f2g_msg.active_sh_degree = gaussian_packet.active_sh_degree 

            f2g_msg.max_sh_degree = gaussian_packet.max_sh_degree
            f2g_msg.xyz = convert_tensor_to_ros_message(gaussian_packet.get_xyz)
            f2g_msg.features = convert_tensor_to_ros_message(gaussian_packet.get_features)
            f2g_msg.scaling = convert_tensor_to_ros_message(gaussian_packet.get_scaling)
            f2g_msg.rotation = convert_tensor_to_ros_message(gaussian_packet.get_rotation)
            f2g_msg.opacity = convert_tensor_to_ros_message(gaussian_packet.get_opacity)
            f2g_msg.normal = convert_tensor_to_ros_message(gaussian_packet.get_normal)

            f2g_msg.n_obs = gaussian_packet.n_obs

            if gaussian_packet.gtcolor is not None:
                f2g_msg.gtcolor = convert_tensor_to_ros_message(gaussian_packet.gtcolor)
            
            if gaussian_packet.gtdepth is not None:
                f2g_msg.gtdepth = convert_tensor_to_ros_message(gaussian_packet.gtdepth)
        

            #f2g_msg.current_frame = self.get_camera_msg_from_viewpoint(gaussian_packet.current_frame)

            f2g_msg.finish = gaussian_packet.finish

        return f2g_msg

    def run(self):
        # set visible devices
        time_recorder = Recorder(args.device_list[0])
        optimization_params = OptimizationParams(parser)
        dataset_params = DatasetParams(parser, sentinel=True)
        map_params = MapParams(parser)

        safe_state(args.quiet)
        optimization_params = optimization_params.extract(args)
        dataset_params = dataset_params.extract(args)
        map_params = map_params.extract(args)

        # Initialize dataset
        dataset = Dataset(
            dataset_params,
            shuffle=False,
            resolution_scales=dataset_params.resolution_scales,
        )

        record_mem = args.record_mem

        gaussian_map = Mapping(args, time_recorder)
        gaussian_map.create_workspace()
        gaussian_tracker = Tracker(args)
        # save config file
        prepare_cfg(args)
        # set time log
        tracker_time_sum = 0
        mapper_time_sum = 0

        # start SLAM
        for frame_id, frame_info in enumerate(dataset.scene_info.train_cameras):
            curr_frame = loadCam(
                dataset_params, frame_id, frame_info, dataset_params.resolution_scales[0]
            )

            print("\n========== curr frame is: %d ==========\n" % frame_id)
            move_to_gpu(curr_frame)
            start_time = time.time()
            # tracker process
            frame_map = gaussian_tracker.map_preprocess(curr_frame, frame_id)
            gaussian_tracker.tracking(curr_frame, frame_map)
            tracker_time = time.time()
            tracker_consume_time = tracker_time - start_time
            time_recorder.update_mean("tracking", tracker_consume_time, 1)

            tracker_time_sum += tracker_consume_time
            print(f"[LOG] tracker cost time: {tracker_time - start_time}")

            mapper_start_time = time.time()

            new_poses = gaussian_tracker.get_new_poses()
            gaussian_map.update_poses(new_poses)
            # mapper process
            gaussian_map.mapping(curr_frame, frame_map, frame_id, optimization_params)

            gaussian_map.get_render_output(curr_frame)
            gaussian_tracker.update_last_status(
                curr_frame,
                gaussian_map.model_map["render_depth"],
                gaussian_map.frame_map["depth_map"],
                gaussian_map.model_map["render_normal"],
                gaussian_map.frame_map["normal_map_w"],
            )
            mapper_time = time.time()
            mapper_consume_time = mapper_time - mapper_start_time
            time_recorder.update_mean("mapping", mapper_consume_time, 1)

            mapper_time_sum += mapper_consume_time
            print(f"[LOG] mapper cost time: {mapper_time - tracker_time}")
            if record_mem:
                time_recorder.watch_gpu()
            # # report eval loss
            # if ((gaussian_map.time + 1) % gaussian_map.save_step == 0) or (
            #     gaussian_map.time == 0
            # ):
            #     eval_frame(
            #         gaussian_map,
            #         curr_frame,
            #         os.path.join(gaussian_map.save_path, "eval_render"),
            #         min_depth=gaussian_map.min_depth,
            #         max_depth=gaussian_map.max_depth,
            #         save_picture=True,
            #         run_pcd=False
            #     )
            #     gaussian_map.save_model(save_data=True)
            gaussian_packet = self.get_gaussian_packet(curr_frame, gaussian_map.global_params_detach)
            self.publish_message_to_gui(gaussian_packet)



            gaussian_map.time += 1
            move_to_cpu(curr_frame)
            torch.cuda.empty_cache()
        print("\n========== main loop finish ==========\n")
        print(
            "[LOG] stable num: {:d}, unstable num: {:d}".format(
                gaussian_map.get_stable_num, gaussian_map.get_unstable_num
            )
        )
        print("[LOG] processed frame: ", gaussian_map.optimize_frames_ids)
        print("[LOG] keyframes: ", gaussian_map.keyframe_ids)
        print("[LOG] mean tracker process time: ", tracker_time_sum / (frame_id + 1))
        print("[LOG] mean mapper process time: ", mapper_time_sum / (frame_id + 1))
        
        new_poses = gaussian_tracker.get_new_poses()
        gaussian_map.update_poses(new_poses)
        gaussian_map.global_optimization(optimization_params, is_end=True)
        # eval_frame(
        #     gaussian_map,
        #     gaussian_map.keyframe_list[-1],
        #     os.path.join(gaussian_map.save_path, "eval_render"),
        #     min_depth=gaussian_map.min_depth,
        #     max_depth=gaussian_map.max_depth,
        #     save_picture=True,
        #     run_pcd=False
        # )
        
        gaussian_map.save_model(save_data=True)
        gaussian_tracker.save_traj(args.save_path)
        time_recorder.cal_fps()
        time_recorder.save(args.save_path)
        gaussian_map.time += 1
        
        if args.pcd_densify:    
            densify_pcd = gaussian_map.stable_pointcloud.densify(1, 30, 5)
            o3d.io.write_point_cloud(
                os.path.join(args.save_path, "save_model", "pcd_densify.ply"), densify_pcd
            )

def spin_thread(node):
    # Spin the node continuously in a separate thread
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)

def main():
    rclpy.init()
    slam_node = SLAM_ROS(args)
    try:
        # Start the spin thread for continuously handling callbacks
        spin_thread_instance = threading.Thread(target=spin_thread, args=(slam_node,))
        spin_thread_instance.start()

        # Run the main logic (this will execute in parallel with message handling)
        slam_node.run()
        
    finally:
        slam_node.destroy_node()
        rclpy.shutdown()
        spin_thread_instance.join()  # Wait for the spin thread to finish

if __name__ == "__main__":
    main()
