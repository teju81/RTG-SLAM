string msg
Camera last_frame
int32 last_frame_id

# gaussian optimize parameters
std_msgs/Float32MultiArray xyz
std_msgs/Float32MultiArray opacity
std_msgs/Float32MultiArray scaling
std_msgs/Float32MultiArray rotation
std_msgs/Float32MultiArray shs #features

# map management paramters
std_msgs/Float32MultiArray normal
std_msgs/Float32MultiArray confidence

#Unknown gaussian parameter
std_msgs/Float32MultiArray radius