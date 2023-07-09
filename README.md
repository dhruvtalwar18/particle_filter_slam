# Particle Filter SLAM


<h1><b> Objective </b></h1>
The objective of this project is to implement Simultaneous Localization and Mapping (SLAM) using a combination of encoder and IMU odometry, 2-D LiDAR scans, and RGBD measurements from a differential-drive robot. The project aims to utilize the odometry and LiDAR measurements to accurately localize the robot and construct a detailed 2-D occupancy grid map of the surrounding environment. Additionally, the project aims to leverage RGBD images to assign colors to the generated 2-D map, providing visual representation of the floor in the mapping process.
<br>

<h1><b> Sensor Description</b></h1>

<b>Encoder<b/>: <br>
The robot's four wheels are equipped with encoders that count rotations at a frequency of 40 Hz. After each reading, the encoder counter is reset. For example, if a rotation corresponds to ` meters traveled, a sequence of five consecutive encoder counts of 0, 1, 0, -2, 3 translates to (0 + 1 + 0 - 2 + 3) = 2 meters traveled for that particular wheel. According to the data sheet, the wheel diameter is 0.254 m, and since there are 360 ticks per revolution, the wheel travels 0.0022 meters per tick. The encoder counts [F R, F L, RR, RL] represent the front-right, front-left, rear-right, and rear-left wheels, respectively. The right wheels cover a distance of (F R + RR)/2 * 0.0022 m, while the left wheels travel (F L + RL)/2 * 0.0022 m.

<b> IMU </b>: <br>
An inertial measurement unit provides linear acceleration and angular velocity data. The IMU data may contain noise due to high-frequency vibrations while the robot is in motion. It is recommended to apply a low-pass filter with a bandwidth of approximately 10 Hz to mitigate measurement noise. Only the yaw rate from the IMU is necessary for the angular velocity in the differential-drive model, and the other IMU measurements are not required.

<b> LiDAR (Hokuyo) </b>: <br>
A horizontal LiDAR with a 270° field of view and a maximum range of 30 m measures distances to obstacles in the environment. Each LiDAR scan consists of 1081 range values. The specific LiDAR sensor used is the Hokuyo UTM-30LX, and its specifications can be found online. The sensor's position relative to the robot body is specified in the provided robot description file. Understanding how to interpret the LiDAR data and convert it from range measurements to (x, y) coordinates in the sensor frame, then to the robot's body frame, and finally to the world frame is crucial.

<b> RGBD Camera (Kinect)</b>: <br>
An RGBD camera captures RGB images and disparity images. The depth camera is located at (0.18, 0.005, 0.36) m relative to the robot's center. It has an orientation with roll 0 rad, pitch 0.36 rad, and yaw 0.021 rad.





<h1><b> 2D Occupancy Grid </b></h1>

<p align="center">
  <img src="https://github.com/dhruvtalwar18/particle_filter_slam/blob/main/Results/Occupancy_grid/Trajectory_map_generation_dataset_20.gif" title="Occupancy Grid Dataset20" style="width: 400px; height: 400px;">
  <br>
  <p align="center">Fig.1 Occupancy Grid Dataset20</p>
</p>


<h1><b> 2-D texture map of the floor </b></h1>

<p align="center">
  <img src="https://github.com/dhruvtalwar18/particle_filter_slam/blob/main/Results/Texture_map/Texture_map_time_generation.gif" title="Texture Map Dataset20" style="width: 400px; height: 400px;">
  <br>
  <p align="center">Fig.2 Texture Map Dataset 20</p>
</p>


<h1><b> Code Implementation </b></h1>

Clone the repository and install the requirements
```
gitclone https://github.com/dhruvtalwar18/particle_filter_slam
cd particle_filter_slam
python3 -m venv venv
pip3 install -r requirements.txt
```
There are codes for 3 different tasks
Trajectory generation: This script runs and saves the trajectory of the robot.
```
python3 trajectory_predict_dead_reckoning.py
```
Map generation: This script runs and saves the occupancy grid from the dataset given
```
python3 single_map_generation.py
```
Texture Mapping: This script runs and saves the texture map of the floor from the dataset given
```
python3 final_particle_filter_texture_map.py
```

<b> Note all scripts need a path to the dataset to run</b>
The dataset can be downloaded from https://drive.google.com/drive/folders/1Fn7YF4u-0bwKGcdKhu76zGfcxNydyXdr?usp=drive_link
The dataset for encoder, LiDAR, and IMU is available in the data folder.
Download the dataRGBD dataset from the provided link and copy the "data" directory to your desired location.

