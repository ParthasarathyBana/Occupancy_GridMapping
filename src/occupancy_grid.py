#!/usr/bin/env python

# Python dependancy in ROS 
import rospy

# Importing math library to perform trigonometric operations
from math import *

# The rosbag file to get sensor data
import rosbag

# Import numpy package
import numpy as np

# The Pillow package for calculating discretizing the map
from PIL import Image

# To visualize the plots we use the cv2 package
import cv2

# Getting an input map for reference
map_original = np.array(Image.open('/home/glaurung/catkin_ws/src/homework2/maps/maze.png'))
# Getting the number of rows and columns of the map
map_rows, map_cols = map_original.shape
# Creating an empty list to store the unoccupied cells on the map
unoccupied_cells = []
# Creating an empty list to store the occupied cells on the map
occupied_cells = []
# Creating a list of the grid structure of the map to store the probabilities of robot's presence in each cell
probability_grid = np.zeros((map_rows, map_cols, 3), 'float64')
# Getting the rosbag file which has data about odom, scan, amcl_pose, teleop of the robot
bag = rosbag.Bag('/home/glaurung/catkin_ws/src/homework2/rosbag/turtlebot_record.bag')

# Function to collect required data from rosbag
def collecting_data(bag):
	# Iterating through all the topics and messages that are present in the rosbag to get the desired topic for use
	for topic, msg, time_stamp in bag.read_messages():
		# Checking for the amcl_pose topic to get the current pose of the robot in the world
		if topic == '/amcl_pose' and msg.header.seq == 0:
			# Getting the initial pose of the robot
			initial_pose = msg.pose.pose
			# Getting the roll, pitch and yaw from the pose obtained from amcl_pose
			old_roll, old_pitch, old_yaw = euler_from_quaternion(initial_pose.orientation.x, initial_pose.orientation.y, initial_pose.orientation.z, initial_pose.orientation.w)
		# Checking for the odom topic to get the current relative pose of the robot in the world	
		if topic == '/odom' and 'initial_pose' in vars():
			# Getting the intial position of the robot relative to the initial amcl pose
			prev_pose = msg.pose.pose
			# Getting the relative roll, pitch and yaw 
			new_roll, new_pitch, new_yaw = euler_from_quaternion(prev_pose.orientation.x, prev_pose.orientation.y, prev_pose.orientation.z, prev_pose.orientation.w)
			# Getting current position and pose of robot
			position_x = prev_pose.position.x + initial_pose.position.x
			position_y = prev_pose.position.y + initial_pose.position.y
			angle_diff = new_yaw - old_yaw
			# Convert the angles into radians
			if angle_diff > np.pi:
				angle_diff = angle_diff - 2*np.pi
			elif angle_diff < np.pi:
				angle_diff = angle_diff + 2*np.pi
			current_pose = angle_diff
		#Checking for the laser scan topic to get the laser scan data 
		if topic == '/scan':
			# Get laser scan range data
			laser_range = msg.ranges
			# Get laser scan angle increment
			angle_increment = msg.angle_increment
			# Get laser scan maximum angle
			angle_max = msg.angle_max
			# Get laser scan minimum angle
			angle_min = msg.angle_min
			# Get laser scan intensities
			intensities = msg.intensities
			# calculate the probability of presence of robot in current cell and update the map based on current pose and position of robot
			probability_grid = occupancy_mapping(position_x, position_y, current_pose, laser_range, angle_increment, angle_max, angle_min, intensities)
	bag.close()

def occupancy_mapping(position_x, position_y, current_pose, laser_range, angle_increment, angle_max, angle_min, intensities):
	global unoccupied_cells
	global occupied_cells
	global map_rows
	global map_cols
	# Map current position of robot onto the grid 
	grid_x, grid_y = map_to_grid(position_x, position_y)
	# For every laser scan in the scope, make a note of all the occupied and unoccupied cells 
	for scan in range(len(laser_range)):
		# Check for presence of obstacle in the scope of laser scan
		if intensities[scan] > 0:
			# Calculate the x and y coordinates of cell which has obstacle
			occupied_cell_x = position_x + laser_range[scan] * cos(current_pose + (angle_min + scan * ((angle_max-angle_min)/len(laser_range))))
			occupied_cell_y = position_y + laser_range[scan] * sin(current_pose + (angle_min + scan * ((angle_max-angle_min)/len(laser_range))))
			# Check for boundary condition of the coordinates
			beyond_map = 1
			if(occupied_cell_x < 0) and (occupied_cell_y < 0):
				beyond_map = 0
			if beyond_map == 0:
				continue
			# Update the grid by mapping the occupied cells' coordinates onto grid
			occupied_cell_x, occupied_cell_y = map_to_grid(occupied_cell_x, occupied_cell_y)
			# Update the list of cells that are unoccupied
			unoccupied_cells = unoccupied_cells + list(bresenham(grid_x, grid_y, occupied_cell_x, occupied_cell_y))
			# Update the list of cells that are occupied
			occupied_cells = occupied_cells + [(occupied_cell_x, occupied_cell_y)]
	# Make an ordered set of both the occupied and unoccupied cell list
	unoccupied_cells = list(set(unoccupied_cells))
	occupied_cells = list(set(occupied_cells))
	# Update the probability grid structure with all the occupied and unoccupied cells
	for each_cell in range(len(occupied_cells)-1):
		x = occupied_cells[each_cell][0]
		y = occupied_cells[each_cell][1]
		probability_grid[x][y] += np.log(0.25/0.75)
	for each_cell in range(len(unoccupied_cells)-1):
		x,y = unoccupied_cells[each_cell]
		probability_grid[x][y] += np.log(0.75/0.25)
	# Visualize the probability grid by plotting it
	prob = 1.0 - 1.0/(1.0 + np.exp(probability_grid))
	window = cv2.namedWindow('Map', cv2.WINDOW_NORMAL)
	cv2.resizeWindow('Map', 700, 700)
	cv2.imshow('Map', prob)
	cv2.waitKey(10)

def euler_from_quaternion(x, y, z, w):

    # Calculating the roll angle
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x**2 + y**2)
    roll = degrees(atan2(t0, t1))

    # Calculating the pitch angle
    t2 = 2.0 * (w * y - z * x)
    t2 = 1.0 if t2 > 1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = degrees(asin(t2))

    # Calculating the yaw angle
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = degrees(atan2(t3, t4))

    # returns the Euler angles
    return roll, pitch, yaw	

def bresenham(x0, y0, x1, y1):
	dx = x1 - x0
	dy = y1 - y0
	xsign = 1 if dx > 0 else -1
	ysign = 1 if dy > 0 else -1
	dx = abs(dx)
	dy = abs(dy)
	if dx > dy:
		xx, xy, yx, yy = xsign, 0, 0, ysign
	else:
		dx, dy = dy, dx
		xx, xy, yx, yy = 0, ysign, xsign, 0
	D = 2*dy - dx
	y = 0
	for x in range(dx + 1):
		m = (x0 + x*xx + y*yx, y0 + x*xy + y*yy)
		if(m[0] < 0) or (m[1] < 0): 
			print((x0, y0), (x1, y1))
		if (m[0] != x1) and (m[1] != y1):
			yield m
		if D >= 0:
			y += 1
			D -= 2*dx
		D += 2*dy

def map_to_grid(position_x, position_y):
	# Map current position_x and position_y coordinates on the grid structure
	grid_x = np.int(position_x*(map_cols - 1)/10.0)	
	grid_y = np.int(position_y*(map_rows - 1)/10.0)
	# Boundary condition to ensure that the positions are mapped on the grid
	if grid_x > map_cols - 1:
		grid_x = map_cols - 1
	if grid_y > map_rows - 1:
		grid_y = map_rows - 1
	# Return the corresponding position coordinates on grid
	return grid_x, grid_y 

if __name__ == '__main__':
	rospy.init_node('Occupancy_Grid')
	rate = rospy.Rate(10)
	rospy.sleep(0.5)
	bag = rosbag.Bag('/home/glaurung/catkin_ws/src/homework2/rosbag/turtlebot_record.bag')
	collecting_data(bag)