import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from mpl_toolkits.mplot3d import Axes3D
import time
import numpy as np
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt

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
  angles = np.arange(-135,135.25,0.25)*np.pi/180.0
  ranges = np.load("/content/drive/MyDrive/ECE276A_PR2/ECE276A_PR2/code/test_ranges.npy")

	

if __name__ == '__main__':
  show_lidar()





import numpy as np
import matplotlib.pyplot as plt
import math

################# Understanding Encoder Data #########################
def edrive_model(encoder_counts, yaw, dt, state):
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
    del_theta = yaw * dt / 2
    x_update = linear_vel * dt * sinc(del_theta) * np.cos(state[2] + del_theta)
    y_update = linear_vel * dt * sinc(del_theta) * np.sin(state[2] + del_theta)
    theta_update = yaw*dt
    
    #x_t = linear_vel * dt * np.cos(state[2])
    #y_t = linear_vel * dt * np.sin(state[2])
    step_update= np.array([x_update, y_update,theta_update])    
    state_updated = state + step_update 
    return state_updated


def sinc(a):
    return np.sin(a)/(0.000000001+a)

def butter_lowpass_filter(data, cutoff=10, fs=22, order=5):
    nyq = fs/2
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff)
    return lfilter(b, a, data)

if __name__=='__main__':

    traj = []
    state_new = np.array([0,0,0])
    state_old = state_new
    yaw_rate = 0
      
    with np.load("/content/drive/MyDrive/ECE276A_PR2/data/Encoders20.npz") as data:
        encoder_counts = data["counts"] # 4 x n encoder counts # (4,4956)
        encoder_stamps = data["time_stamps"] # encoder time stamps # 4956
        
    with np.load("/content/drive/MyDrive/ECE276A_PR2/data/Imu20.npz") as data:
        imu_angular_velocity = data["angular_velocity"] # angular velocity in rad/sec # (3,12187)
        imu_linear_acceleration = data["linear_acceleration"] # Accelerations in gs (gravity acceleration scaling) # (3,12187)
        imu_stamps = data["time_stamps"]  # acquisition times of the imu measurements # 12187

    encoder_counts_1 = encoder_counts.T
    imu_angular_velocity_changed = imu_angular_velocity.T
    time_stamp = imu_stamps
    noisy = imu_angular_velocity_changed[:,2]
    filtered_IMU_yaw = butter_lowpass_filter(noisy, 10, 22, order=5)
    
    counter = [0, 0, 0]
    end_flag = False

    while not end_flag:
      if counter[0] >= encoder_stamps.shape[0]-1 or counter[1] >= imu_stamps.shape[0]-1:
        end_flag = True
      else:
        if encoder_stamps[counter[0]] <= imu_stamps[counter[1]]:
            counter[0] += 1
            dt = encoder_stamps[counter[0]+1] - encoder_stamps[counter[0]]     
            state_old = state_new
            state_new = edrive_model(encoder_counts_1[counter[0]+1], yaw_rate, dt, state_old)
            traj.append(state_new)

        elif imu_stamps[counter[1]] <= encoder_stamps[counter[0]]:
            counter[1] += 1
            #yaw_rate = imu_angular_velocity[:,counter[1]][2]   
            yaw_rate = filtered_IMU_yaw[counter[1]]

    


# Show the plot
plt.show()                
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlabel("X position")
ax.set_ylabel("Y position")
ax.set_title("Trajectory of particles Dataset 20")
ax.legend()
trajectory = np.asarray(traj)
plt.plot(trajectory[:,0],trajectory[:,1])

###############################################################################

import numpy as np
import matplotlib.pyplot as plt
import math

################# Understanding Encoder Data #########################
def edrive_model(encoder_counts, yaw, dt, state):
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
    del_theta = yaw * dt / 2
    x_update = linear_vel * dt * sinc(del_theta) * np.cos(state[2] + del_theta)
    y_update = linear_vel * dt * sinc(del_theta) * np.sin(state[2] + del_theta)
    theta_update = yaw*dt
    
    #x_t = linear_vel * dt * np.cos(state[2])
    #y_t = linear_vel * dt * np.sin(state[2])
    step_update= np.array([x_update, y_update,theta_update])    
    state_updated = state + step_update 
    return state_updated


def sinc(a):
    return np.sin(a)/(1e-9+a)

def butter_lowpass_filter(data, cutoff=10, fs=22, order=5):
    nyq = fs/2
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff)
    return lfilter(b, a, data)

if __name__=='__main__':
    #state = np.array([0,0,0])
    #state_0 = state
    particle_number = 20 # number of particles
    yaw = 0
    N = 20
    trajectory_state = [np.empty((0, 3)) for _ in range(N)]    
    counter = [0,0,0]
    yaw_rate = 0
    end_flag = False   
    state_new = np.zeros((N, 3)) 
    state_old = state_new

      
    with np.load("/content/drive/MyDrive/ECE276A_PR2/data/Encoders20.npz") as data:
        encoder_counts = data["counts"] # 4 x n encoder counts # (4,4956)
        encoder_stamps = data["time_stamps"] # encoder time stamps # 4956
        
    with np.load("/content/drive/MyDrive/ECE276A_PR2/data/Imu20.npz") as data:
        imu_angular_velocity = data["angular_velocity"] # angular velocity in rad/sec # (3,12187)
        imu_linear_acceleration = data["linear_acceleration"] # Accelerations in gs (gravity acceleration scaling) # (3,12187)
        imu_stamps = data["time_stamps"]  # acquisition times of the imu measurements # 12187

    encoder_counts_1 = encoder_counts.T
    imu_angular_velocity_changed = imu_angular_velocity.T
    time_stamp = imu_stamps
    noisy = imu_angular_velocity_changed[:,2]
    filtered_IMU_yaw = butter_lowpass_filter(noisy, 10, 22, order=5)  


    while not end_flag:
      if counter[0] >= encoder_stamps.shape[0]-1 or counter[1] >= imu_stamps.shape[0]-1:
        end_flag = True
      else:
        if encoder_stamps[counter[0]] <= imu_stamps[counter[1]]:
            counter[0] += 1
            dt = encoder_stamps[counter[0]+1] - encoder_stamps[counter[0]]  
            for i in range(particle_number):
              state_old[i] = state_new[i]
              state_old[i] = state_old[i] + np.random.normal(loc=0.0, scale=0.001, size=state_new[i].shape)
              state_new[i] = edrive_model(encoder_counts_1[counter[0]+1], yaw, dt, state_old[i])
              state_new[i] = state_new[i] 
              trajectory_state[i] = np.vstack([trajectory_state[i], state_new[i]])

        elif imu_stamps[counter[1]] <= encoder_stamps[counter[0]]:
            counter[1] += 1
            yaw = imu_angular_velocity[:,counter[1]][2]   
            #yaw = filtered_IMU_yaw[counter[1]]


# Initialize the figure
fig, ax = plt.subplots(figsize=(15, 15))
# Plot the trajectory of each particle
for i in range(particle_number):
    ax.plot(trajectory_state[i][:, 0], trajectory_state[i][:, 1], label=f"Particle {i+1}")
# Add labels and legend
ax.set_xlabel("X position")
ax.set_ylabel("Y position")
ax.set_title("Trajectory of particles 20")
ax.legend()
# Show the plot
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import math

################# Understanding Encoder Data #########################
def edrive_model(encoder_counts, yaw, dt, state):
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
    del_theta = yaw * dt / 2
    x_update = linear_vel * dt * sinc(del_theta) * np.cos(state[2] + del_theta)
    y_update = linear_vel * dt * sinc(del_theta) * np.sin(state[2] + del_theta)
    theta_update = yaw*dt
    
    #x_t = linear_vel * dt * np.cos(state[2])
    #y_t = linear_vel * dt * np.sin(state[2])
    step_update= np.array([x_update, y_update,theta_update])    
    state_updated = state + step_update 
    return state_updated


def sinc(a):
    return np.sin(a)/(1e-9+a)

def butter_lowpass_filter(data, cutoff=10, fs=22, order=5):
    nyq = fs/2
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff)
    return lfilter(b, a, data)

if __name__=='__main__':
    #state = np.array([0,0,0])
    #state_0 = state
    particle_number = 20 # number of particles
    yaw = 0
    N = 20
    trajectory_state = [np.empty((0, 3)) for _ in range(N)]    
    counter = [0,0,0]
    yaw_rate = 0
    end_flag = False   
    state_new = np.zeros((N, 3)) 
    state_old = state_new

      
    with np.load("/content/drive/MyDrive/ECE276A_PR2/data/Encoders20.npz") as data:
        encoder_counts = data["counts"] # 4 x n encoder counts # (4,4956)
        encoder_stamps = data["time_stamps"] # encoder time stamps # 4956
        
    with np.load("/content/drive/MyDrive/ECE276A_PR2/data/Imu20.npz") as data:
        imu_angular_velocity = data["angular_velocity"] # angular velocity in rad/sec # (3,12187)
        imu_linear_acceleration = data["linear_acceleration"] # Accelerations in gs (gravity acceleration scaling) # (3,12187)
        imu_stamps = data["time_stamps"]  # acquisition times of the imu measurements # 12187

    encoder_counts_1 = encoder_counts.T
    imu_angular_velocity_changed = imu_angular_velocity.T
    time_stamp = imu_stamps
    noisy = imu_angular_velocity_changed[:,2]
    filtered_IMU_yaw = butter_lowpass_filter(noisy, 10, 22, order=5)  


    while not end_flag:
      if counter[0] >= encoder_stamps.shape[0]-1 or counter[1] >= imu_stamps.shape[0]-1:
        end_flag = True
      else:
        if encoder_stamps[counter[0]] <= imu_stamps[counter[1]]:
            counter[0] += 1
            dt = encoder_stamps[counter[0]+1] - encoder_stamps[counter[0]]  
            for i in range(particle_number):
              state_old[i] = state_new[i]
              state_old[i] = state_old[i] + np.random.normal(loc=0.0, scale=0.001, size=state_new[i].shape)
              state_new[i] = edrive_model(encoder_counts_1[counter[0]+1], yaw, dt, state_old[i])
              state_new[i] = state_new[i] 
              trajectory_state[i] = np.vstack([trajectory_state[i], state_new[i]])

        elif imu_stamps[counter[1]] <= encoder_stamps[counter[0]]:
            counter[1] += 1
            yaw = imu_angular_velocity[:,counter[1]][2]   
            #yaw = filtered_IMU_yaw[counter[1]]


# Initialize the figure
fig, ax = plt.subplots(figsize=(15, 15))
# Plot the trajectory of each particle
for i in range(particle_number):
    ax.plot(trajectory_state[i][:, 0], trajectory_state[i][:, 1], label=f"Particle {i+1}")
# Add labels and legend
ax.set_xlabel("X position")
ax.set_ylabel("Y position")
ax.set_title("Trajectory of particles 20")
ax.legend()
# Show the plot
plt.show()

"""This code makes N particles, initalizes them then applies the motion model to it with noise on every step and gets the trajectories"""