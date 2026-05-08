# Ackermann model in ros2 aws small warehouse
A simple achermann model with warehouse environment that could use to test SLAM, map and planning.
## 1. build ROS file
```
colcon build
source install/setup.bash
```
## 2. run gazebo simulation
```
ros2 launch car_description gazebo.launch.py 
```
## 3. run ros2_control 
```
ros2 topic pub /rear_wheel_velocity_controller/commands std_msgs/msg/Float64MultiArray "{data: [2.0, 2.0]}"
ros2 topic pub /front_steering_position_controller/commands std_msgs/msg/Float64MultiArray "{data: [0.3, 0.3]}"
```
<img width="2560" height="1600" alt="Screenshot from 2026-05-02 21-30-34" src="https://github.com/user-attachments/assets/8e833e6f-f8ec-4eec-8abb-91962f18b14e" />

Use simple keyboard to control the car
(w: forward, s: backward, a: turn left, d: turn right)
```
ros2 run controller keyboard_controller
```

## 4. run rtab-map with RGBD camera
install rtabmap
```
sudo apt update
sudo apt install ros-jazzy-rtabmap-ros
```
run rtabmap
```
ros2 launch rtabmap_launch rtabmap.launch.py \
  rgb_topic:=/depth_camera/image \
  depth_topic:=/depth_camera/depth_image \
  camera_info_topic:=/depth_camera/camera_info \
  frame_id:=base_link \
  odom_topic:=/rtabmap/odom \
  approx_sync:=true \
  rgbd_sync:=true \
  visual_odometry:=true \
  rtabmap_viz:=true \
  rviz:=true
```
<img width="2560" height="1600" alt="Screenshot from 2026-05-06 18-02-19" src="https://github.com/user-attachments/assets/aebfa014-2adc-40f0-aa0a-991f8b787589" />

## Compare path with gazebo ground truth
First bag your odometry topic 
```
ros2 bag record \
/rtabmap/odom \
/ground_truth/odom \
/tf \
/tf_static
```
and run the script and play your ros bag to compare the path from SLAM with Gazebo ground truth
```
python3 compare_odom_paths.py
```
<img width="2560" height="1600" alt="Screenshot from 2026-05-08 16-09-52" src="https://github.com/user-attachments/assets/ebc0ada8-3f23-4678-b255-cf4a47fc98a5" />




