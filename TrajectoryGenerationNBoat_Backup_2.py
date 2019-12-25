import numpy as np 
import pandas as pd 
import random
import matplotlib.pyplot as plt
import math
import cv2
import csv

class TrajectoryGenerationNBoat:
	"""
	This class is responsible to generate the NBoat trajectory.

	Attributes
	----------------
	distance 	: 	int[][]
		matrix of distance generated to the specific final point.
	rewards		:	int[][]
		matrix of rewards generated to the specific final point.
	startPoint	: (int, int)
		point of start
	endPoint	: (int, int)
		point of end
	matrix_size	: int
		size of matrix (matrix_size x matrix_size)

	Methods
	---------------
	move(direction, blocks)
		Performs the movement of the agent due the action desired
	is_blocked(blocks, point)
		Used on the generation of reward matrix to identify where is placed the borders and obstacles in the map

	"""

	reward = None
	positionCol = None
	positionRow = None

	#Actions declarations
	ACTION_UP = "Up"
	ACTION_DOWN = "Down"
	ACTION_LEFT = "Left"
	ACTION_RIGHT = "Right"
	ACTION_UP_LEFT = "UpLeft"
	ACTION_UP_RIGHT = "UpRight"
	ACTION_DOWN_LEFT = "DownLeft"
	ACTION_DOWN_RIGHT = "DownRight"

	def __init__(self, startPoint, endPoint, matrix_size):
		"""
		Parameters
		-------------
		distance 	: 	int[][]
			matrix of distance generated to the specific final point.
		rewards		:	int[][]
			matrix of rewards generated to the specific final point.
		startPoint	: (int, int)
			point of start
		endPoint	: (int, int)
			point of end
		matrix_size	: int
			size of matrix (matrix_size x matrix_size)
		"""

		self.startPoint = startPoint
		self.endPoint = endPoint
		self.positionRow = startPoint[1]
		self.positionCol = startPoint[0]
		self.endPointRow = endPoint[1]
		self.endPointCol = endPoint[0]
		self.matrix_size = matrix_size

	def setDistace(self, distance):
		self.distance = distance

	def setRewards(self, rewards):
		self.rewards = rewards

	def setWindDirection(self, wind_direction):
		self.wind_direction = wind_direction

	def define_action_reward(self):
		#Wind direction
		action = 0 #all_directions
		#only_diagonal = False
		
		wind_direction = self.wind_direction #degrees:: use 225 to dead zone
		print("Wind Direction: {}".format(wind_direction))

		#boat_direction
		theta = math.atan2((self.endPointRow - self.positionRow), (self.endPointCol - self.positionCol))
		boat_direction = math.degrees(theta)
		boat_direction = boat_direction + 360 if boat_direction < 0 else boat_direction
		#boat_direction = (final_row - start_row)^2 
		print("Boat direction: {}".format(round(boat_direction,2)))

		#verify if boat direction is opposed to wind
		interval = wind_direction - boat_direction
		interval = interval + 360 if interval < 0 else interval
		
		angle_dict = {"Right":0, "UpRight":45, "Up":90, "UpLeft":135, "Left":180, "DownLeft":225, "Down":270, "DownRight":315}

		print("interval: {}".format(interval))
		if  interval >= 140 and interval <= 290:
			print("Dead Zone")
			interval = wind_direction - 180
			interval = interval + 360 if interval < 0 else interval
			range_wind = (interval + 40, interval - 40)

			rem = []
			for angle in angle_dict:
				for value in range(range_wind[1], range_wind[0]):
					if value == angle_dict[angle]:
						print("There is {}".format(angle))
						rem.append(angle)
						break

			for r in rem:
				del angle_dict[r]

		index = []
		for angle in angle_dict:
			index.append(angle)
					
		print(angle_dict)

		
		qtable = pd.DataFrame(100, index=index, columns=self.generate_columns_qtable())

			#if (boat_direction <= 30 and boat_direction >= 0) or (boat_direction >= 330) or (boat_direction <= 120 and boat_direction >= 60) or (boat_direction <= 210 and boat_direction >= 150) or (boat_direction <= 300 and boat_direction >= 240):
			#	only_diagonal = True
			#	print("Only Diagonal")
			#	action = 1 #Diagonal Actions
			#else:
			#	only_diagonal = False
			#	action = 2 #Standard Actions
			#	print("Only Simple Actions")

		return qtable

	def move(self, direction, blocks):
		"""
		Parameters
		--------------
		directions	:	string
			Direction to move the agent
		blocks		:	int[(int,int)]
			Positions of blocks points

		Returns
		--------------
		(reward, end) 
			a tuple with "reward" value for that state and a boolean value to indicate if the agent achieved the end. 
		"""
		reward = 0
		end = False
		distance_before = self.distance[self.positionCol][self.positionRow]

		if direction == TrajectoryGenerationNBoat.ACTION_UP:
			self.positionRow +=1
		elif direction == TrajectoryGenerationNBoat.ACTION_DOWN:
			self.positionRow -=1
		elif direction == TrajectoryGenerationNBoat.ACTION_LEFT:
			self.positionCol -= 1
		elif direction == TrajectoryGenerationNBoat.ACTION_RIGHT:
			self.positionCol += 1
		elif direction == TrajectoryGenerationNBoat.ACTION_UP_LEFT:
			self.positionRow +=1
			self.positionCol -= 1
		elif direction == TrajectoryGenerationNBoat.ACTION_UP_RIGHT:
			self.positionRow +=1
			self.positionCol += 1
		elif direction == TrajectoryGenerationNBoat.ACTION_DOWN_LEFT:
			self.positionRow -=1
			self.positionCol -= 1
		elif direction == TrajectoryGenerationNBoat.ACTION_DOWN_RIGHT:
			self.positionRow -=1
			self.positionCol += 1

		if (self.is_blocked(blocks, (self.positionCol, self.positionRow))):
			reward = -1000
			end = True
		elif self.positionRow < 1 or self.positionRow > self.matrix_size or self.positionCol < 1 or self.positionCol > self.matrix_size:
			end = True
			reward = -1000	
		elif self.positionCol == self.endPointCol and self.positionRow == self.endPointRow:
			end = True
			reward = self.rewards[self.positionCol][self.positionRow].item()
		else:
			end = False
			if distance_before.item() < self.distance[self.positionCol][self.positionRow].item():
				reward = -1
			else:
				reward = self.rewards[self.positionCol][self.positionRow].item()

		return (reward, end)

	def is_blocked(self, blocks, point):
		"""This function is used on the generation of reward matrix to identify where is placed the borders and obstacles in the map.

		Parameters
		--------------
		blocks	:	int[(int,int)]
			A list of points that indicates the positions of the obstacles and borders
		point 	: 	(int,int)
			Point that must be verified inside blocks list

		Returns
		--------------
		True 
			if point exists into blocks
		False 
			if not

		"""
		for b in blocks:
			if(b[0]==point[0] and b[1]==point[1]):
				return True
		return False

	def there_is(self, already_generated, point, nrow, ncol):
		for ag in already_generated:
			if point[0] < 1 or point[0] > nrow or point[1] < 1 or point[1] > ncol:
				return True
			if(ag[0] == point[0] and ag[1] == point[1]):
				return True

		return False

	def generate_reward_distance_map(self, frow, fcol, only_diagonal=False):		
		nrow = self.matrix_size
		ncol = self.matrix_size

		end_generation = False
		distance = pd.DataFrame(1000, index=np.arange(1,self.matrix_size), columns=np.arange(1,self.matrix_size))
		already_generated = []
		positionCol, positionRow = fcol, frow
		

		while not end_generation:
			if (positionRow == frow and positionCol == fcol):
				distance.loc[positionCol,positionRow] = 0
				positionRow = 1000
				already_generated.append((fcol, frow))

			for ag in already_generated:
				(i, j) = ag
				if only_diagonal:
					if(not self.there_is(already_generated,(i+1,j+1), nrow, ncol)):
						distance.loc[i+1,j+1] = distance.loc[i,j] + 1
						already_generated.append((i+1, j+1))
					if(not self.there_is(already_generated, (i+1,j-1), nrow, ncol)):
						distance.loc[i+1,j-1] = distance.loc[i,j] + 1
						already_generated.append((i+1, j-1))
					if(not self.there_is(already_generated, (i-1,j+1), nrow, ncol)):
						distance.loc[i-1,j+1] = distance.loc[i,j] + 1
						already_generated.append((i-1, j+1))
					if(not self.there_is(already_generated, (i-1,j-1), nrow, ncol)):
						distance.loc[i-1,j-1] = distance.loc[i,j] + 1
						already_generated.append((i-1, j-1))
					if(not self.there_is(already_generated,(i+1,j), nrow, ncol)):
						distance.loc[i+1,j] = 1000
						already_generated.append((i+1, j))
					if(not self.there_is(already_generated, (i,j+1), nrow, ncol)):
						distance.loc[i,j+1] = 1000
						already_generated.append((i, j+1))
					if(not self.there_is(already_generated, (i-1,j), nrow, ncol)):
						distance.loc[i-1,j] = 1000
						already_generated.append((i-1, j))
					if(not self.there_is(already_generated, (i,j-1), nrow, ncol)):
						distance.loc[i,j-1] = 1000
						already_generated.append((i, j-1))
				else:
					if(not self.there_is(already_generated,(i+1,j), nrow, ncol)):
						distance.loc[i+1,j] = distance.loc[i,j] + 1
						already_generated.append((i+1, j))
					if(not self.there_is(already_generated, (i,j+1), nrow, ncol)):
						distance.loc[i,j+1] = distance.loc[i,j] + 1
						already_generated.append((i, j+1))
					if(not self.there_is(already_generated, (i-1,j), nrow, ncol)):
						distance.loc[i-1,j] = distance.loc[i,j] + 1
						already_generated.append((i-1, j))
					if(not self.there_is(already_generated, (i,j-1), nrow, ncol)):
						distance.loc[i,j-1] = distance.loc[i,j] + 1
						already_generated.append((i, j-1))

			if len(already_generated) == nrow*ncol:
				end_generation = True

		rewards = pd.DataFrame(100, index=np.arange(1,nrow), columns=np.arange(1,ncol))
		max_distance = 0
		for i in range(1,self.matrix_size):
			for j in range(1, self.matrix_size):
				if(distance.loc[i,j] > max_distance):
					max_distance = distance.loc[i,j]
		
		for i in range(1,self.matrix_size):
			for j in range(1, self.matrix_size):
				rewards.loc[i,j] = abs(distance.loc[i,j] - max_distance)
		print(rewards)
		#print(distance)
		return rewards, distance


	def create_map(self, qtable, blocks=None):
		nrow = self.matrix_size
		ncol = self.matrix_size
		data_index_x = []
		for i in range(1,nrow):
			data_index_x.append(i)
		map = pd.DataFrame(2, index=data_index_x, columns=data_index_x)

		self.positionRow = self.startPoint[1]
		self.positionCol = self.startPoint[0]

		end_step = False
		old_current_state = 0
		repeat = 0
		
		if not blocks == None:
			for b in blocks:
				map.loc[b[1],b[0]] = 3

		step = 0
		while not end_step:
			current_state_ = self.positionRow*100+ self.positionCol*10 + self.positionRow
			step =+ 1
			direction_ = qtable[current_state_].idxmax()
			print("Step: {}> Direction: {}".format(step, direction_))
			#if random.random() < 0.2:
			#	direction_ = qtable[current_state_].nlargest(2).index.all()
			
			map.loc[self.positionRow, self.positionCol] = 1
			if direction_== TrajectoryGenerationNBoat.ACTION_UP:
				self.positionRow += 1
			elif direction_== TrajectoryGenerationNBoat.ACTION_DOWN:
				self.positionRow -= 1
			elif direction_ == TrajectoryGenerationNBoat.ACTION_LEFT:
				self.positionCol -= 1
			elif direction_== TrajectoryGenerationNBoat.ACTION_RIGHT:
				self.positionCol += 1
			elif direction_== TrajectoryGenerationNBoat.ACTION_UP_LEFT:
				self.positionRow +=1
				self.positionCol -= 1
			elif direction_== TrajectoryGenerationNBoat.ACTION_UP_RIGHT:
				self.positionRow +=1
				self.positionCol += 1
			elif direction_ == TrajectoryGenerationNBoat.ACTION_DOWN_LEFT:
				self.positionRow -=1
				self.positionCol -= 1
			elif direction_ == TrajectoryGenerationNBoat.ACTION_DOWN_RIGHT:
				self.positionRow -=1
				self.positionCol += 1

			if self.positionCol == self.endPointCol and self.positionRow == self.endPointRow:
				end_step = True
				map.loc[self.positionRow, self.positionCol] = 12

			if(old_current_state == current_state_):
				print("No way!")
				break
			else:
				if repeat == 1:
					old_current_state = current_state_
					repeat = -1
				repeat += 1

		map.loc[self.startPoint[1], self.startPoint[0]] = 9
		map.loc[self.endPointRow, self.endPointCol] = 16

		
		return map

	def generate_columns_qtable(self):
		cols = []
		for i in range(1,self.matrix_size):
			for j in range(1,self.matrix_size):
				cols.append(j*100+i*10 + j)

		return cols

	def show_trajectory(self, map_):
		fig, ax = plt.subplots()
		ax.matshow(map_, cmap=plt.cm.tab20c)
		xpoints = []
		ypoints = []
		points = []

		for i in range(1,self.matrix_size):
			for j in range(1,self.matrix_size):
				if(map_.loc[i,j]==16) or (map_.loc[i,j] == 9) or (map_.loc[i,j]==1):
					xpoints.append(j-1)
					ypoints.append(i-1)
		#map = pd.DataFrame(2, index=xpoints, columns=ypoints)
		#print(map)
		ax.set_xlim([-1,self.matrix_size-1])
		ax.set_ylim([-1,self.matrix_size-1])
		plt.grid(b=True, axis='both')

		major_ticks = np.arange(-1, self.matrix_size, 1)
		ax.set_xticks(major_ticks)
		major_ticks_y = np.arange(-1, self.matrix_size, 1)
		ax.set_yticks(major_ticks_y)
		ax.grid(which='both')

		ax.plot(xpoints,ypoints, 'ow')
		#df = pd.read_csv('data_out_obstacle.csv')
		#xpoints = []
		#ypoints = []
		#cont = 0
		#for i in range(1,len(df.index)):
	#		if cont == 0:
		#		xpoints.append(df.loc[i][0])
		#		ypoints.append(df.loc[i][1])
		#	cont+=1
		#	if cont == 100:
		#		cont = 0
		#print(len(xpoints))
		#plt.plot(xpoints,ypoints, 'og')
		plt.show()

		#ax.annotate('WD', xy =(2,12), xytext = (2,15),arrowprops=dict(facecolor='black', shrink=0.05))
		plt.show()
		return xpoints, ypoints

	def train(self, learning_rate, discount, random_explore, horizon, blocks):
		qtable = self.define_action_reward()
		only_diagonal=False
		rewards, distance = self.generate_reward_distance_map(self.endPoint[0], self.endPoint[1], only_diagonal)
		self.setDistace(distance)
		self.setRewards(rewards)

		if(self.is_blocked(blocks, self.startPoint) or self.is_blocked(blocks, self.endPoint)):
			print("Please, verify the start point position and end point position! They are placed on blocked square!")
			exit(0)

		#if(actions==0):
		#	qtable = pd.DataFrame(100, index=['Up','Down','Left','Right','UpLeft', 'UpRight', 'DownLeft', 'DownRight'], columns=self.generate_columns_qtable())
		#elif (actions==1):
		#	qtable = pd.DataFrame(100, index=['UpLeft', 'UpRight', 'DownLeft', 'DownRight'], columns=self.generate_columns_qtable())
		#else:
		#	qtable = pd.DataFrame(100, index=['Up','Down','Right','Left'], columns=self.generate_columns_qtable())
		
		for i in range(horizon):
			print("Step: {}".format(i), end='\r')
			nboatrl = TrajectoryGenerationNBoat(self.startPoint, self.endPoint, self.matrix_size)

			nboatrl.setDistace(distance)
			nboatrl.setRewards(rewards)
			end_step = False
			while not end_step:
				current_state = nboatrl.positionRow*100+ (nboatrl.positionCol*10)+nboatrl.positionRow
				max_reward_action = qtable[current_state].idxmax()

				if random.random() < random_explore:
					max_reward_action = qtable.index[random.randint(0,len(qtable.index)-1)]

				reward, end_step = nboatrl.move(max_reward_action, blocks)

				if end_step:
					qtable.loc[max_reward_action, current_state] = reward

				else:
					opimtal_future_value = qtable[nboatrl.positionRow*100 + (nboatrl.positionCol*10)+nboatrl.positionRow].max()
					discounted_opimtal_future_value = discount * opimtal_future_value
					learned_value = reward + discounted_opimtal_future_value
					qtable.loc[max_reward_action, current_state] = (1 - learning_rate) * qtable[current_state][max_reward_action] + learning_rate * learned_value
					#learning_rate =- 1/horizon
										#print(max_reward_action, current_state, (1 - learning_rate) * qtable[current_state][max_reward_action] + learning_rate * learned_value)
			#print("------------------")

		return qtable

	def get_map_blocks(self, img):
		img = cv2.imread(img,cv2.IMREAD_COLOR)
		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		cv2.imwrite('new_image.jpg', gray)
		new_image = []
		width, height, channel = img.shape
		#gray = gray[5:width-5, 5:height-5]
		n = self.matrix_size
		square_size = round(width/n)
		point_position = round(square_size - square_size/2)
		#print(point_position)
		col, row = 1,1

		matrix = []
		for i in range(width - point_position, point_position, -square_size):
			matrix_row = []
			for j in range(point_position, height, square_size):
				gray_value = gray[i][j]
					
				if (gray_value > 60 and gray_value < 90):
					matrix_row.append((row,col))
				row += 1
			col += 1
			row = 1
			matrix.append(matrix_row)
		
		xpoints = []
		ypoints = []
		blocks = []

		for row in matrix:
			for i in row:
				xpoints.append(i[0])
				ypoints.append(i[1])
				blocks.append((i[0],i[1]))
		#plt.plot(xpoints,ypoints, 'o')

		#plt.show()

		return blocks


	def compute_gps_coordinates(self, blocks):
		xpoints, ypoints = blocks
		#Lat and Long positions
		lat = [-6.011959, -6.011959, -6.032615, -6.032615]
		lon = [-35.208616, -35.187960, -35.187960, -35.208616]

		point_dif_lat = (lat[2] - lat[1])/self.matrix_size
		point_dif_lon = (lon[0] - lon[1])/self.matrix_size
		new_x_points =[]
		new_y_points =[]
		for i in range(0,len(xpoints)):
			new_x_points.append((xpoints[i]+1)*(-point_dif_lat)+lon[3])
			new_y_points.append((ypoints[i]+1)*(-point_dif_lon)+lat[3])

		#for i in range(0, len(new_x_points)):
		#	print(new_y_points[i], new_x_points[i])

		with open('cartesian_file.csv', mode='w') as file:
			coordenates_writer = csv.writer(file, delimiter=',')
			coordenates_writer.writerow(['x','y'])
			for i in range(0, len(new_x_points)):
				coordenates_writer.writerow([xpoints[i], ypoints[i]])
			print("Coordinates saved in /cartesian_file.csv")

		with open('coordinates_file.csv', mode='w') as file:
			coordenates_writer = csv.writer(file, delimiter=',')
			coordenates_writer.writerow(['latitude','longitude'])
			for i in range(0, len(new_x_points)):
				coordenates_writer.writerow([new_y_points[i], new_x_points[i]])
			print("Coordinates saved in /coordinates_file.csv")
