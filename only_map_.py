import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from mpl_toolkits.mplot3d import Axes3D
import time
import numpy as np
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
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
def map_init():  
    MAP = {}
    MAP['res']   = 0.02 #meters
    MAP['xmin']  = -30  #meters
    MAP['ymin']  = -30
    MAP['xmax']  =  30
    MAP['ymax']  =  30 
    MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
    MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1)) #cells
    MAP['map'] = np.log(0.5)*np.ones((MAP['sizex'],MAP['sizey']))
    return MAP

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
    return np.sin(a)/(0.00000001+a)

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
    filtered_angles, filtered_ranges = filter_lidar_data(lidar_angle_min, lidar_angle_max, lidar_angle_increment, lidar_ranges,lidar_range_min, lidar_range_max,t)

    body_x = filtered_ranges * np.cos(filtered_angles)
    body_y = filtered_ranges * np.sin(filtered_angles)
    world_x = body_x * np.cos(state[2]) - body_y * np.sin(state[2]) + state[0]
    world_y = body_x * np.sin(state[2]) + body_y * np.cos(state[2]) + state[1]

    # Get cell coordinates of particle and sensor beams
    particle_cell_x = pos_to_cell_x(state[0], MAP)
    particle_cell_y = pos_to_cell_y(state[1], MAP)
    sensor_cell_x = pos_to_cell_x(world_x, MAP)
    sensor_cell_y = pos_to_cell_y(world_y, MAP)

    # Update occupancy map
    for i in range(sensor_cell_x.shape[0]):
        cell_x, cell_y = bresenham2D(particle_cell_x, particle_cell_y, sensor_cell_x[i], sensor_cell_y[i])
        valid_indices = np.logical_and(cell_x > 1, cell_y > 1)
        MAP['map'][cell_x[valid_indices].astype(np.int16), cell_y[valid_indices].astype(np.int16)] += np.log(4)
        valid_indices = np.logical_and(sensor_cell_x > 1, sensor_cell_y > 1)
        MAP['map'][sensor_cell_x[valid_indices], sensor_cell_y[valid_indices]] += -np.log(4)
    return MAP


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

###############################################################################
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

    MAP=map_init()    
    x_range = np.arange(-0.2, 0.2 + 0.05, 0.05)
    y_range = np.arange(-0.2, 0.2 + 0.05, 0.05)
    x_im = np.arange(MAP['xmin'], MAP['xmax'] + MAP['res'], MAP['res'])  # physical x-positions of each pixel of the map
    y_im = np.arange(MAP['ymin'], MAP['ymax'] + MAP['res'], MAP['res'])  # physical y-positions of each pixel of the map
    
    state_new = np.array([0,0,0])
    state_old = state_new
    yaw_rate = 0

    trajectory_map = []
    encoder_counts_1 = encoder_counts.T
    imu_angular_velocity_changed = imu_angular_velocity.T
    time_stamp = imu_stamps
    noisy = imu_angular_velocity_changed[:,2]
    #filtered_IMU_yaw = butter_lowpass_filter(noisy, 10, 22, order=5)

    counter = [0, 0, 0]
    end_flag = False

    while not end_flag:
      if counter[0] >= encoder_stamps.shape[0]-1 or counter[1] >= imu_stamps.shape[0]-1 or counter[2] >= lidar_stamps.shape[0]-1:
        end_flag = True
      else:
        if encoder_stamps[counter[0]] <= imu_stamps[counter[1]] and encoder_stamps[counter[0]] <= lidar_stamps[counter[2]]:
            counter[0] += 1
            dt = encoder_stamps[counter[0]+1] - encoder_stamps[counter[0]]     
            state_old = state_new
            state_new = edrive_model(encoder_counts_1[counter[0]+1], yaw_rate, dt, state_old)
        elif imu_stamps[counter[1]] <= encoder_stamps[counter[0]] and imu_stamps[counter[1]] <= lidar_stamps[counter[2]]:
            counter[1] += 1
            yaw_rate = imu_angular_velocity[:,counter[1]][2]   
        else:
            counter[2] += 1
            MAP = map_generation(counter[2],state_new)
            if counter[2]%200==0 :
              final_map1 = np.exp(MAP['map'])/(1+np.exp(MAP['map']))
              final1 = final_map1 
              plt.figure(figsize=(6, 6))    
              plt.imshow(final1,cmap='gray')
              plt.show()


    final_map = np.exp(MAP['map'])/(1+np.exp(MAP['map']))
    final = final_map 
    plt.figure(figsize=(15, 15))    
    plt.imshow(final,cmap='gray')
    plt.show()