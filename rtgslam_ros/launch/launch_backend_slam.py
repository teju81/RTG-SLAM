from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='rtgslam_ros',  # Name of the package containing the node
            executable='slam_backend',  # Name of the node executable
            name='tracker_backend_node',  # Optional: Name to give the node
            output='screen',  # Output of the node (can be 'screen' or 'log')
        )
    ])