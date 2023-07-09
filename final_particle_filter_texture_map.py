import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from mpl_toolkits.mplot3d import Axes3D
import time
import numpy as np
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
from PIL import Image

def tic():
  return time.time()
def toc(tstart, name="Operation"):
  print('%s took: %s sec.\n' % (name,(time.time() - tstart)))


def mapCorrelation(im, x_im, y_im, vp, xs, ys):
  '''
  INPUT 
  im              the map 
  x_im,y_im       physical x,y positions of the grid map cells
  vp[0:2,:]       occupied x,y positions from range sensor (in physical unit)  
  xs,ys           physical x,y,positions you want to evaluate "correlation" 

  OUTPUT 
  c               sum of the cell values of all the positions hit by range sensor
  '''
  nx = im.shape[0]
  ny = im.shape[1]
  xmin = x_im[0]
  xmax = x_im[-1]
  xresolution = (xmax-xmin)/(nx-1)
  ymin = y_im[0]
  ymax = y_im[-1]
  yresolution = (ymax-ymin)/(ny-1)
  nxs = xs.size
  nys = ys.size
  cpr = np.zeros((nxs, nys))
  for jy in range(0,nys):
    y1 = vp[1,:] + ys[jy] # 1 x 1076
    iy = np.int16(np.round((y1-ymin)/yresolution))
    for jx in range(0,nxs):
      x1 = vp[0,:] + xs[jx] # 1 x 1076
      ix = np.int16(np.round((x1-xmin)/xresolution))
      valid = np.logical_and( np.logical_and((iy >=0), (iy < ny)), \
			                        np.logical_and((ix >=0), (ix < nx)))
      cpr[jx,jy] = np.sum(im[ix[valid],iy[valid]])
  return cpr


def bresenham2D(sx, sy, ex, ey):
  '''
  Bresenham's ray tracing algorithm in 2D.
  Inputs:
	  (sx, sy)	start point of ray
	  (ex, ey)	end point of ray
  '''
  sx = int(round(sx))
  sy = int(round(sy))
  ex = int(round(ex))
  ey = int(round(ey))
  dx = abs(ex-sx)
  dy = abs(ey-sy)
  steep = abs(dy)>abs(dx)
  if steep:
    dx,dy = dy,dx # swap 

  if dy == 0:
    q = np.zeros((dx+1,1))
  else:
    q = np.append(0,np.greater_equal(np.diff(np.mod(np.arange( np.floor(dx/2), -dy*dx+np.floor(dx/2)-1,-dy),dx)),0))
  if steep:
    if sy <= ey:
      y = np.arange(sy,ey+1)
    else:
      y = np.arange(sy,ey-1,-1)
    if sx <= ex:
      x = sx + np.cumsum(q)
    else:
      x = sx - np.cumsum(q)
  else:
    if sx <= ex:
      x = np.arange(sx,ex+1)
    else:
      x = np.arange(sx,ex-1,-1)
    if sy <= ey:
      y = sy + np.cumsum(q)
    else:
      y = sy - np.cumsum(q)
  return np.vstack((x,y))
    

def test_bresenham2D():
  import time
  sx = 0
  sy = 1
  print("Testing bresenham2D...")
  r1 = bresenham2D(sx, sy, 10, 5)
  r1_ex = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10],[1,1,2,2,3,3,3,4,4,5,5]])
  r2 = bresenham2D(sx, sy, 9, 6)
  r2_ex = np.array([[0,1,2,3,4,5,6,7,8,9],[1,2,2,3,3,4,4,5,5,6]])	
  if np.logical_and(np.sum(r1 == r1_ex) == np.size(r1_ex),np.sum(r2 == r2_ex) == np.size(r2_ex)):
    print("...Test passed.")
  else:
    print("...Test failed.")

  # Timing for 1000 random rays
  num_rep = 1000
  start_time = time.time()
  for i in range(0,num_rep):
	  x,y = bresenham2D(sx, sy, 500, 200)
  print("1000 raytraces: --- %s seconds ---" % (time.time() - start_time))

def test_mapCorrelation():
  angles = np.arange(-135,135.25,0.25)*np.pi/180.0
  ranges = np.load("/content/drive/MyDrive/ECE276A_PR2/ECE276A_PR2/code/test_ranges.npy")

  # take valid indices
  indValid = np.logical_and((ranges < 30),(ranges> 0.1))
  ranges = ranges[indValid]
  angles = angles[indValid]

  # init MAP
  MAP = {}
  MAP['res']   = 0.05 #meters
  MAP['xmin']  = -20  #meters
  MAP['ymin']  = -20
  MAP['xmax']  =  20
  MAP['ymax']  =  20 
  MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
  MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
  MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.int8) #DATA TYPE: char or int8
  

  
  # xy position in the sensor frame
  xs0 = ranges*np.cos(angles)
  ys0 = ranges*np.sin(angles)
  
  # convert position in the map frame here 
  Y = np.stack((xs0,ys0))
  
  # convert from meters to cells
  xis = np.ceil((xs0 - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
  yis = np.ceil((ys0 - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
  
  # build an arbitrary map 
  indGood = np.logical_and(np.logical_and(np.logical_and((xis > 1), (yis > 1)), (xis < MAP['sizex'])), (yis < MAP['sizey']))
  MAP['map'][xis[indGood[0]],yis[indGood[0]]]=1
      
  x_im = np.arange(MAP['xmin'],MAP['xmax']+MAP['res'],MAP['res']) #x-positions of each pixel of the map
  y_im = np.arange(MAP['ymin'],MAP['ymax']+MAP['res'],MAP['res']) #y-positions of each pixel of the map

  x_range = np.arange(-0.2,0.2+0.05,0.05)
  y_range = np.arange(-0.2,0.2+0.05,0.05)


  
  print("Testing map_correlation with {}x{} cells".format(MAP['sizex'],MAP['sizey']))
  ts = tic()
  c = mapCorrelation(MAP['map'],x_im,y_im,Y,x_range,y_range)
  toc(ts,"Map Correlation")

  c_ex = np.array([[3,4,8,162,270,132,18,1,0],
		  [25  ,1   ,8   ,201  ,307 ,109 ,5  ,1   ,3],
		  [314 ,198 ,91  ,263  ,366 ,73  ,5  ,6   ,6],
		  [130 ,267 ,360 ,660  ,606 ,87  ,17 ,15  ,9],
		  [17  ,28  ,95  ,618  ,668 ,370 ,271,136 ,30],
		  [9   ,10  ,64  ,404  ,229 ,90  ,205,308 ,323],
		  [5   ,16  ,101 ,360  ,152 ,5   ,1  ,24  ,102],
		  [7   ,30  ,131 ,309  ,105 ,8   ,4  ,4   ,2],
		  [16  ,55  ,138 ,274  ,75  ,11  ,6  ,6   ,3]])
    
  if np.sum(c==c_ex) == np.size(c_ex):
	  print("...Test passed.")
  else:
    print("...Test failed. Close figures to continue tests.")	

  #plot original lidar points
  fig1 = plt.figure()
  plt.plot(xs0,ys0,'.k')
  plt.xlabel("x")
  plt.ylabel("y")
  plt.title("Laser reading")
  plt.axis('equal')
  
  #plot map
  fig2 = plt.figure()
  plt.imshow(MAP['map'],cmap="hot");
  plt.title("Occupancy grid map")
  
  #plot correlation
  fig3 = plt.figure()
  ax3 = fig3.gca(projection='3d')
  X, Y = np.meshgrid(np.arange(0,9), np.arange(0,9))
  ax3.plot_surface(X,Y,c,linewidth=0,cmap=plt.cm.jet, antialiased=False,rstride=1, cstride=1)
  plt.title("Correlation coefficient map")  
  plt.show()
  
  
def show_lidar():
  print("OK")

if __name__ == '__main__':
  show_lidar()


############################################### CODE STARTS ####################

import numpy as np
import matplotlib.pyplot as plt
import math

def map_init():  
    MAP = {}
    MAP['res']   = 0.05 #meters
    MAP['xmin']  = -35  #meters
    MAP['ymin']  = -35
    MAP['xmax']  =  35
    MAP['ymax']  =  35 
    MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
    MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1)) #cells
    MAP['map'] = np.log(0.5)*np.ones((MAP['sizex'],MAP['sizey']))
    MAP['trajectory'] = np.log(0.5)*np.ones((MAP['sizex'],MAP['sizey']))
    MAP['texture'] = np.zeros([MAP['sizex'],MAP['sizey'],3])
    return MAP

def edrive_model(encoder_counts, yaw, dt, state,P):
    TICKS_PER_REV = 360
    WHEEL_DIAMETER = 0.254
    DIST_PER_TICK = (WHEEL_DIAMETER * math.pi) / TICKS_PER_REV
    # Calculate distance traveled by each wheel
    dist_fr = encoder_counts[0] * DIST_PER_TICK
    dist_fl = encoder_counts[1] * DIST_PER_TICK
    dist_rr = encoder_counts[2] * DIST_PER_TICK
    dist_rl = encoder_counts[3] * DIST_PER_TICK
    # Calculate total distance traveled by right and left wheels
    dist_right = (dist_fr + dist_rr) / 2
    dist_left = (dist_fl + dist_rl) / 2
    # Calculate linear and angular velocity of the robot
    linear_vel = (dist_right + dist_left) / (2 * dt)
    angular_vel = (dist_right - dist_left) / (2 * dt)  
    d = yaw * dt / 2
    for i in range(len(P)):
      state = P[i,:]
      x_t = linear_vel * dt * sinc(d) * np.cos(state[2] + d)
      y_t = linear_vel * dt * sinc(d) * np.sin(state[2] + d)
      #x_t = linear_vel * dt * np.cos(state[2])
      #y_t = linear_vel * dt * np.sin(state[2])
      u = np.array([x_t, y_t, yaw * dt])
      state_updated = state + u
      P[i,:] = state_updated +  np.random.normal(0, 0.0001, P[i,:].shape)
      #P = P + u + 
    return state_updated, P

def sinc(a):
    return np.sin(a)/(0.00000001+a)
def softmax(a):
    return np.exp(a-a.max()) / np.exp(a-a.max(axis=0)).sum()


def trajectory_conversion(X_old,Y_old,X_new,Y_new,MAP):
    x_new = pos_to_cell_x(X_new, MAP)
    y_new = pos_to_cell_y(Y_new, MAP)
    x_prev = pos_to_cell_x(X_old, MAP)
    y_prev = pos_to_cell_y(Y_old, MAP)
    x_i, y_i = bresenham2D(x_prev,y_prev,x_new,y_new).astype(int)
    return x_i, y_i

def filter_lidar_data(lidar_angle_min, lidar_angle_max, lidar_angle_increment, lidar_ranges,lidar_range_min, lidar_range_max,t):
    angles = np.arange(lidar_angle_min, lidar_angle_max + lidar_angle_increment - 0.001, lidar_angle_increment)
    ranges = lidar_ranges[:, t]
    valid = np.logical_and(ranges < lidar_range_max, ranges > lidar_range_min)
    angles = angles[valid]
    ranges = ranges[valid]
    return angles, ranges


def map_generation(t, state):
    x_p = state[0]
    y_p = state[1]
    theta_p = state[2]
    filtered_angles, filtered_ranges = filter_lidar_data(lidar_angle_min, lidar_angle_max, lidar_angle_increment, lidar_ranges,lidar_range_min, lidar_range_max,t)

    x_b = filtered_ranges * np.cos(filtered_angles)
    y_b = filtered_ranges * np.sin(filtered_angles)
    x_w = x_b * np.cos(theta_p) - y_b * np.sin(theta_p) + x_p
    y_w = x_b * np.sin(theta_p) + y_b * np.cos(theta_p) + y_p

    # Get cell coordinates of particle and sensor beams
    particle_cell_x = pos_to_cell_x(x_p, MAP)
    particle_cell_y = pos_to_cell_y(y_p, MAP)
    sensor_cell_x = pos_to_cell_x(x_w, MAP)
    sensor_cell_y = pos_to_cell_y(y_w, MAP)

    # Update occupancy map
    for i in range(sensor_cell_x.shape[0]):
        cell_x, cell_y = bresenham2D(particle_cell_x, particle_cell_y, sensor_cell_x[i], sensor_cell_y[i])
        valid_indices = np.logical_and(cell_x > 1, cell_y > 1)
        MAP['map'][cell_x[valid_indices].astype(np.int16), cell_y[valid_indices].astype(np.int16)] += np.log(4)

    valid_indices = np.logical_and(sensor_cell_x > 1, sensor_cell_y > 1)
    MAP['map'][sensor_cell_x[valid_indices], sensor_cell_y[valid_indices]] -= np.log(4)
    MAP['map'][MAP['map'] > 0] = 0

    return MAP

def calculate_correlation_scores(num, P, x_b, y_b, MAP, x_im, y_im, x_range, y_range):
  for i in range(num):
    x_world = x_b * np.cos(P[i,:][2]) - y_b * np.sin(P[i,:][2]) + P[i,:][0]
    y_world = x_b * np.sin(P[i,:][[2]]) + np.cos(P[i,:][2]) + P[i,:][1]
    vp = np.stack((x_world, y_world))
    corr[i] = np.max(mapCorrelation(MAP['map'], x_im, y_im, vp, x_range, y_range))
  return corr

def update(t, P, MAP, num,Weights):
    filtered_angles, filtered_ranges = filter_lidar_data(lidar_angle_min, lidar_angle_max, lidar_angle_increment, lidar_ranges,lidar_range_min, lidar_range_max,t)   
    # from lidar frame to world frame
    x_b = filtered_ranges * np.cos(filtered_angles)
    y_b = filtered_ranges * np.sin(filtered_angles)
    corr = calculate_correlation_scores(num, P, x_b, y_b, MAP, x_im, y_im, x_range, y_range)
    cors = W * np.array(corr)   
    Weights = softmax(cors)
    max_index = np.argmax(Weights)
    state = P[max_index,:]
   
    if 1/(Weights**2).sum() < N_eff:
        Weights, P = resampling(Weights, P, num)
    return P, state, Weights


def resampling(weights, poses, num_particles):
    # normalize the weights
    weights /= np.sum(weights)    
    # initialize new particle array and weight array
    new_poses = np.zeros((num_particles, poses.shape[1]))
    new_weights = np.ones(num_particles) / num_particles
    
    # perform the resampling algorithm
    index = int(np.random.rand() * num_particles)
    beta = 0
    max_weight = np.max(weights)
    
    for i in range(num_particles):
        beta += np.random.rand() * 2.0 * max_weight
        while beta > weights[index]:
            beta -= weights[index]
            index = (index + 1) % num_particles
        new_poses[i, :] = poses[index, :]
        new_weights[i] = new_weights[i] * weights[index]
    
    # normalize the new weights
    new_weights /= np.sum(new_weights)
    
    return new_weights, new_poses

def butter_lowpass_filter(data, cutoff=10, fs=22, order=5):
    nyq = fs/2
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff)
    return lfilter(b, a, data)
################################################################################
def calculate_depth_mask(disparity, rows, cols,dd):
    depth_mask = np.logical_and(dd>0,dd<5)
    grid = np.indices((rows,cols))
    pixels = rows*cols
    x = grid[0]
    y = grid[1]
    x.resize((1,pixels))
    y.resize((1,pixels))
    image_index = np.vstack((x,y))
    image_frame = np.vstack((x,y,np.ones(pixels)))    
    return depth_mask, image_index, image_frame


def apply_depth_mask(image_index, image_frame, disparity, dd, depth_mask):
    image_index = image_index[:,depth_mask]
    image_frame = image_frame[:,depth_mask]
    disparity = disparity[depth_mask]
    dd = dd[depth_mask]
    
    return image_index, image_frame, disparity, dd


def transform_camera_frame(image_frame, dd, K_matrix):
    # Tranform camera frame
    camera_frame = np.linalg.inv(K_matrix) * image_frame
    camera_frame = dd * np.asarray(camera_frame)
    camera_frame = np.vstack((camera_frame, np.ones(camera_frame.shape[1])))
    roll_angle = 0
    pitch_angle = 0.36
    yaw_angle = 0.021

    rotation_o = np.array([[0, -1, 0],
                            [0, 0, -1],
                             [1, 0, 0]])

    rotation_x = np.array([[1, 0, 0],
                        [0, math.cos(roll_angle), -math.sin(roll_angle)],
            [0, math.sin(roll_angle), math.cos(roll_angle)]])

    rotation_y = np.array([[math.cos(pitch_angle), 0, math.sin(pitch_angle)],
                                  [0, 1, 0],
                                  [-math.sin(pitch_angle), 0, math.cos(pitch_angle)]])

    rotation_z = np.array([[math.cos(yaw_angle), -math.sin(yaw_angle), 0],
    [math.sin(yaw_angle), math.cos(yaw_angle), 0],
                        [0, 0, 1]])

    rotation_cam_body = rotation_z.dot(rotation_y.dot(rotation_x))
    position_cam_body = np.array([[0.18], [0.005], [0.36]])
    pose_cb = np.identity(4)
    pose_cb[0:3, 0:3] = rotation_o.dot(rotation_cam_body.T)
    pose_cb[0:3, -1] = -rotation_o.dot(rotation_cam_body.T).dot(position_cam_body).reshape(3)
    pose_cb_inv = np.linalg.inv(pose_cb)
    pose_cb_inv= np.matrix(pose_cb_inv)

    Transformation_b = pose_cb_inv
    

    # Convert to body frame
    body_frame =  Transformation_b * camera_frame

    return body_frame

def transform_body_frame(thresh_img,body_frame,  image_index,disparity, state_new_):
    mask = thresh_img[2,:] < 1 
    thresh_img = body_frame[:, mask]
    thresh_ind = image_index[:, mask]
    d_thres = disparity[mask]
    Transformation_world = np.matrix([[np.cos(state_new_[2]), -np.sin(state_new_[2]),  0, state_new_[0]],
                      [np.sin(state_new_[2]),  np.cos(state_new_[2]),  0, state_new_[1]],
                      [0,                0,                1, 0],
                      [0,                0,                0, 1]])
    world_frame = Transformation_world * thresh_img
    
    return thresh_img, thresh_ind, d_thres, world_frame


def pos_to_cell_x(pos, MAP):
    x = pos
    xmin  = MAP['xmin']
    res = MAP['res']
    x_cell = np.ceil((x - xmin) / res).astype(np.int16)-1
    return x_cell
def pos_to_cell_y(pos, MAP):
    y = pos
    ymin  = MAP['ymin']
    res = MAP['res']
    y_cell = np.ceil((y - ymin) / res).astype(np.int16)-1
    return y_cell

################################################################################
if __name__=='__main__':
     
    with np.load("/content/drive/MyDrive/ECE276A_PR2/data/Encoders21.npz") as data:
        encoder_counts = data["counts"] # 4 x n encoder counts # (4,4956)
        encoder_stamps = data["time_stamps"] # encoder time stamps # 4956
        
    with np.load("/content/drive/MyDrive/ECE276A_PR2/data/Imu21.npz") as data:
        imu_angular_velocity = data["angular_velocity"] # angular velocity in rad/sec # (3,12187)
        imu_linear_acceleration = data["linear_acceleration"] # Accelerations in gs (gravity acceleration scaling) # (3,12187)
        imu_stamps = data["time_stamps"]  # acquisition times of the imu measurements # 12187
    with np.load("/content/drive/MyDrive/ECE276A_PR2/data/Hokuyo21.npz") as data:
        lidar_angle_min = data["angle_min"] # start angle of the scan [rad] # -2.35
        lidar_angle_max = data["angle_max"] # end angle of the scan [rad] # 2.35
        lidar_angle_increment = data["angle_increment"] # angular distance between measurements [rad] # 0.00436
        lidar_range_min = data["range_min"] # minimum range value [m] # 0.1
        lidar_range_max = data["range_max"] # maximum range value [m] # 30
        lidar_ranges = data["ranges"]       # range data [m] (Note: values < range_min or > range_max should be discarded) # (1081,4962)
        lidar_stamps = data["time_stamps"]  # acquisition times of the lidar scans # 4962

    with np.load("/content/drive/MyDrive/ECE276A_PR2/data/Kinect21.npz") as data:
        disp_stamps = data["disparity_time_stamps"] # acquisition times of the disparity images # 2407
        rgb_stamps = data["rgb_time_stamps"] # acquisition times of the rgb images # 2289

    MAP=map_init()    
    x_range = np.arange(-0.2, 0.2 + 0.05, 0.05)
    y_range = np.arange(-0.2, 0.2 + 0.05, 0.05)
    x_im = np.arange(MAP['xmin'], MAP['xmax'] + MAP['res'], MAP['res'])  # physical x-positions of each pixel of the map
    y_im = np.arange(MAP['ymin'], MAP['ymax'] + MAP['res'], MAP['res'])  # physical y-positions of each pixel of the map
    
    
    # particle filter
    particle_number  = 50
    N_eff = 0.05
    W = np.ones(particle_number) / particle_number # weight
    P = np.zeros([particle_number,3]) # pose for particles
    corr = np.zeros(particle_number)
    state_new = np.array([0,0,0])
    state_old = state_new
    yaw_rate = 0
   
    trajectory_map = []
    encoder_counts_1 = encoder_counts.T
    imu_angular_velocity_changed = imu_angular_velocity.T
    time_stamp = imu_stamps
    noisy = imu_angular_velocity_changed[:,2]
    filtered_IMU_yaw = butter_lowpass_filter(noisy, 10, 22, order=5)
    dis_image=np.array([]) 

    counter = [0, 0, 0,0,0]
    end_flag = False

    while not end_flag:
      if counter[0] >= encoder_stamps.shape[0]-1 or counter[1] >= imu_stamps.shape[0]-1 or counter[2] >= lidar_stamps.shape[0]-1 or counter[3]>=disp_stamps.shape[0]-1 or counter[4]>=rgb_stamps.shape[0]-1:
        end_flag = True
      else:
        if encoder_stamps[counter[0]] <= imu_stamps[counter[1]] and encoder_stamps[counter[0]] <= lidar_stamps[counter[2]] and encoder_stamps[counter[0]]<=disp_stamps[counter[3]] and encoder_stamps[counter[0]]<= rgb_stamps[counter[4]]:
            counter[0] += 1
            dt = encoder_stamps[counter[0]+1] - encoder_stamps[counter[0]]     
            state_old = state_new
            state_new, P = edrive_model(encoder_counts_1[counter[0]+1], yaw_rate, dt, state_old,P)


        elif imu_stamps[counter[1]] <= encoder_stamps[counter[0]] and imu_stamps[counter[1]] <= lidar_stamps[counter[2]] and imu_stamps[counter[1]]<=disp_stamps[counter[3]] and imu_stamps[counter[1]]<= rgb_stamps[counter[4]]:
            counter[1] += 1
            #yaw_rate = imu_angular_velocity[:,counter[1]][2]   
            yaw_rate = filtered_IMU_yaw[counter[1]]
        elif  lidar_stamps[counter[2]] <= encoder_stamps[counter[0]] and lidar_stamps[counter[2]] <= imu_stamps[counter[1]] and lidar_stamps[counter[2]]<=disp_stamps[counter[3]] and lidar_stamps[counter[2]]<= rgb_stamps[counter[4]]:

            counter[2] += 1
            P,state_new_,W = update(counter[2],P,MAP,particle_number,W)
            MAP = map_generation(counter[2],state_new_)
            x,y = trajectory_conversion(state_old[0],state_old[1],state_new_[0],state_new[1],MAP) 
            MAP['trajectory'][x,y] = -3000
            trajectory_map.append(state_new)
            if counter[2]%200 == 0:
              print(counter[2],'lidar stamps')
              final_map = np.exp(MAP['trajectory'] + MAP['map'] + 0.6)
                #final_map = np.exp(MAP['map'])
              plt.figure(figsize=(8, 8))    
              final_map[final_map<0.75] = 0.3
              final_map[final_map>0.90] = 1
              plt.imshow(final_map,cmap='gray')
              plt.clim(0,1)
              plt.show()
     



        elif disp_stamps[counter[3]] <= encoder_stamps[counter[0]] and disp_stamps[counter[3]] <= imu_stamps[counter[1]] and disp_stamps[counter[3]]<=lidar_stamps[counter[2]] and disp_stamps[counter[3]]<= rgb_stamps[counter[4]]:
            counter[3] +=1
            disp_image = Image.open('/content/drive/MyDrive/ECE276A_PR2/data/dataRGBD/Disparity21/disparity21_'+str(counter[3])+'.png')
            disp = np.array(disp_image.getdata(),np.uint16)
            disp = disp.reshape(disp_image.size[1]*disp_image.size[0])
        else:
            counter[4] +=1
            # read RGB
            rgb = plt.imread('/content/drive/MyDrive/ECE276A_PR2/data/dataRGBD/RGB21/rgb21_'+str(counter[4])+'.png')
            rows,cols,channel = rgb.shape
            disparity = disp


            dd = 1.03/(-0.00304*disparity + 3.31)
            depth_mask_updated, image_index_updated, image_frame_updated = calculate_depth_mask(disparity, rows, cols,dd)
            image_index_1, image_frame_1, disparity_1, dd_1 = apply_depth_mask(image_index_updated,  image_frame_updated, disparity, dd, depth_mask_updated) 
            K_matrix = np.matrix([[585.05108211, 0           , 242.94140713],
                                  [0           , 585.05108211, 315.83800193],
                           [0           , 0           , 1           ]])
            body_frame = transform_camera_frame(image_frame_1, dd_1, K_matrix)
            thresh_img = np.asarray(body_frame)
            thresh_img_1, thresh_ind, d_thres, world_frame_1=transform_body_frame(thresh_img,body_frame,  image_index_1,disparity_1, state_new_)
               
            rgbi = []
            rgbj = []
            dd = (-0.00304 * d_thres + 3.31)
            depth = 1.03 / dd
            rgbi = np.round((thresh_ind[0, :] * 526.37 + dd * (-4.5 * 1750.46) + 19276) / 585.051).astype(int)
            rgbj = np.round((thresh_ind[1, :] * 526.37 + 16662) / 585.051).astype(int)


            rgbi_arr = np.array(rgbi)
            rgbj_arr = np.array(rgbj)

            # Define boolean masks to filter values that are within the image dimensions
            rgbi_mask = (rgbi_arr > 0) & (rgbi_arr <= rows)
            rgbj_mask = (rgbj_arr > 0) & (rgbj_arr <= cols)
            mask = rgbi_mask & rgbj_mask
            # Create a numpy array containing the row and column indices of the thresholded RGB values
            rgb_indices = np.vstack((rgbi_arr, rgbj_arr))

            # Apply the mask to the RGB indices and world frame arrays
            thresholded_rgb_indices = rgb_indices[:, mask]
            world_frame_masked = world_frame_1[:, mask]                            

            
            # Convert world frame positions to texture cell indices
            x_texture = pos_to_cell_x(world_frame_masked[0,:],MAP)
            y_texture = pos_to_cell_y(world_frame_masked[1,:],MAP)

            # Copy RGB values from input image to texture map at corresponding cell indices
            rgb_indices = np.stack((thresholded_rgb_indices[0,:], thresholded_rgb_indices[1,:]), axis=-1)
            rgb_values = rgb[rgb_indices[:,0], rgb_indices[:,1], :]
            MAP['texture'][x_texture, y_texture, :] = rgb_values

            if counter[4]%100 ==0:
              print("RGB",counter[4])
              plt.figure(figsize=(6, 6)) 
              plt.imshow(MAP['texture'])

            

    plt.imshow(MAP['texture'])