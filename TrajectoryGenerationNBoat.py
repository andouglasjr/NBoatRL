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

	def __init__(self, startPoint, endPoint, matrix_size, test_name):
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
		self.test_name = test_name

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
		theta = math.atan2(-(self.endPointRow - self.startPoint[1]), (self.endPointCol - self.startPoint[0]))
		boat_direction = math.degrees(theta)
		#boat_direction = boat_direction + 360 if boat_direction < 0 else boat_direction
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
			self.positionRow -=1
		elif direction == TrajectoryGenerationNBoat.ACTION_DOWN:
			self.positionRow +=1
		elif direction == TrajectoryGenerationNBoat.ACTION_LEFT:
			self.positionCol -= 1
		elif direction == TrajectoryGenerationNBoat.ACTION_RIGHT:
			self.positionCol += 1
		elif direction == TrajectoryGenerationNBoat.ACTION_UP_LEFT:
			self.positionRow -=1
			self.positionCol -= 1
		elif direction == TrajectoryGenerationNBoat.ACTION_UP_RIGHT:
			self.positionRow -=1
			self.positionCol += 1
		elif direction == TrajectoryGenerationNBoat.ACTION_DOWN_LEFT:
			self.positionRow +=1
			self.positionCol -= 1
		elif direction == TrajectoryGenerationNBoat.ACTION_DOWN_RIGHT:
			self.positionRow +=1
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

	def there_is(self, already_generated, point, matrix_size):
		for ag in already_generated:
			if point[0] < 1 or point[0] > matrix_size or point[1] < 1 or point[1] > matrix_size:
				return True
			if(ag[0] == point[0] and ag[1] == point[1]):
				return True

		return False

	def generate_reward_distance_map(self, end_point, only_diagonal=False):		
		end_generation = False
		distance = pd.DataFrame(1000, index=np.arange(1,self.matrix_size+1), columns=np.arange(1,self.matrix_size+1))
		already_generated = []
		positionCol, positionRow = end_point[1], end_point[0]
		distance.loc[positionCol,positionRow] = 0
		already_generated.append((end_point[1], end_point[0]))	
		
		while not end_generation:	

			for ag in already_generated:
				(i, j) = ag	
			
				if(not self.there_is(already_generated,(i+1,j), self.matrix_size)):
					distance.loc[i+1,j] = distance.loc[i,j] + 1
					already_generated.append((i+1, j))
				if(not self.there_is(already_generated, (i,j+1), self.matrix_size)):
					distance.loc[i,j+1] = distance.loc[i,j] + 1
					already_generated.append((i, j+1))
				if(not self.there_is(already_generated, (i-1,j), self.matrix_size)):
					distance.loc[i-1,j] = distance.loc[i,j] + 1
					already_generated.append((i-1, j))
				if(not self.there_is(already_generated, (i,j-1), self.matrix_size)):
					distance.loc[i,j-1] = distance.loc[i,j] + 1
					already_generated.append((i, j-1))

			if len(already_generated) == self.matrix_size*self.matrix_size:
				end_generation = True

		rewards = pd.DataFrame(100, index=np.arange(1,self.matrix_size+1), columns=np.arange(1,self.matrix_size+1))
		max_distance = 0

		for i in distance.columns:
			for j in distance.index:
				if(distance.loc[i,j] > max_distance):
					max_distance = distance.loc[i,j]
		
		for i in distance.columns:
			for j in distance.index:
				rewards.loc[i,j] = abs(distance.loc[i,j] - max_distance)

		rewards.loc[end_point[1], end_point[0]] = 1000
		#print(rewards)
		#print(distance)
		return rewards, distance


	def create_map(self, qtable, blocks=None):
		map = pd.DataFrame(2, index=range(1,self.matrix_size+1), columns=range(1,self.matrix_size+1))
		self.xpoints = []
		self.ypoints = []
		#print(map)
		self.positionRow = self.startPoint[1]
		self.positionCol = self.startPoint[0]

		end_step = False
		old_current_state = 0
		repeat = 0
		
		if not blocks == None:
			for b in blocks:
				map.loc[b[1],b[0]] = 3

		step = 0
		fig, ax = plt.subplots(figsize=(10,10))
		
		while not end_step:
			current_state_ = self.positionRow*100+ self.positionCol*10 + self.positionRow
			self.xpoints.append(self.positionCol - 1)
			self.ypoints.append(self.positionRow - 1)
			step += 1
			direction_ = qtable[current_state_].idxmax()
			#print("Step: {}> Current State: {}, Direction: {} ".format(step, current_state_, direction_))
			
			map.loc[self.positionRow, self.positionCol] = 1
			if direction_== TrajectoryGenerationNBoat.ACTION_UP:
				self.positionRow -= 1
			elif direction_== TrajectoryGenerationNBoat.ACTION_DOWN:
				self.positionRow += 1
			elif direction_ == TrajectoryGenerationNBoat.ACTION_LEFT:
				self.positionCol -= 1
			elif direction_== TrajectoryGenerationNBoat.ACTION_RIGHT:
				self.positionCol += 1
			elif direction_== TrajectoryGenerationNBoat.ACTION_UP_LEFT:
				self.positionRow -=1
				self.positionCol -= 1
			elif direction_== TrajectoryGenerationNBoat.ACTION_UP_RIGHT:
				self.positionRow -=1
				self.positionCol += 1
			elif direction_ == TrajectoryGenerationNBoat.ACTION_DOWN_LEFT:
				self.positionRow +=1
				self.positionCol -= 1
			elif direction_ == TrajectoryGenerationNBoat.ACTION_DOWN_RIGHT:
				self.positionRow +=1
				self.positionCol += 1

			if self.positionCol == self.endPointCol and self.positionRow == self.endPointRow:
				end_step = True
				print("End path found!")
				#map.loc[self.positionCol, self.positionRow] = 12


			if(old_current_state == current_state_):
				print("No way!")
				break
			else:
				if repeat == 1:
					old_current_state = current_state_
					repeat = -1
				repeat += 1
			#ax.matshow(map, cmap=plt.cm.tab20c)
		map.loc[self.startPoint[1], self.startPoint[0]] = 9
		map.loc[self.endPointRow, self.endPointCol] = 16
		self.xpoints.append(self.endPointCol - 1)
		self.ypoints.append(self.endPointRow - 1)
		
		return map

	def generate_columns_qtable(self):
		cols = []
		for i in range(1,self.matrix_size+1):
			for j in range(1,self.matrix_size+1):
				cols.append(j*100+i*10 + j)

		return cols

	def show_tracetory_from_file(self, csv_points, control_points=None, blocks = None, map = None):
		fig, ax = plt.subplots(figsize=(20,20))	
		xpoints = []
		ypoints = []
		if map is None:
			map = pd.DataFrame(2, index=range(1,self.matrix_size+1), columns=range(1,self.matrix_size+1))
			
			
			if not blocks == None:
				for b in blocks:
					map.loc[b[1],b[0]] = 3

			step = 0

			df = pd.read_csv(csv_points)

			xpoints = []
			ypoints = []
			cont = 0
			for i in range(0,len(df.index)):
				ypoints.append(df.loc[i][0] + 1)
				xpoints.append(df.loc[i][1] + 1)
			
			for i in range(1, len(df.index) - 1):
				map.loc[xpoints[i], ypoints[i]] = 1	
			
			if(self.endPointRow > self.startPoint[1]):
				map.loc[xpoints[0], ypoints[0]] = 9
				map.loc[xpoints[-1], ypoints[-1]] = 16
			else:
				map.loc[xpoints[-1], ypoints[-1]] = 9
				map.loc[xpoints[0], ypoints[0]] = 16

			#fig, ax = plt.subplots(figsize=(15,15))

			xpoints_ = []
			ypoints_ = []
			#points = []
			for i in range(0, len(xpoints)):
				xpoints_.append(xpoints[i]-1)
				ypoints_.append(ypoints[i]-1)

			xpoints = xpoints_
			ypoints = ypoints_
		else:
			ypoints = self.xpoints
			xpoints = self.ypoints

		ax.matshow(map, cmap=plt.cm.tab20c)
		ax.set_xlim([-0.5,self.matrix_size-1.5])
		ax.set_ylim([-0.5,self.matrix_size-1.5])
		ax.grid(b=True, axis='both')

		major_ticks = np.arange(0.5, self.matrix_size-1, 1)
		ax.set_xticks([])
		major_ticks_y = np.arange(-0.5, self.matrix_size-1, 1)
		ax.set_yticks([])
		ax.grid(which='both')

		ax.plot(ypoints,xpoints, '--ow', label="Generated Path")

		bbox = dict(boxstyle="round", fc="0.8")
		arrowprops = dict(
		    arrowstyle = "->",
		    connectionstyle = "angle,angleA=0,angleB=90,rad=10")

		offset = 40
		if map is None:
			if(self.endPointRow > self.startPoint[1]):
				ax.annotate('Start Point',
		            (ypoints[0],xpoints[0]), xytext=(-2*offset, -offset), textcoords='offset points',
		            bbox=bbox, arrowprops=arrowprops)
				ax.annotate('End Point',
		            (ypoints[-1],xpoints[-1]), xytext=(offset, offset), textcoords='offset points',
		            bbox=bbox, arrowprops=arrowprops)
			else:
				ax.annotate('Start Point',
		            (ypoints[-1],xpoints[-1]), xytext=(offset, offset), textcoords='offset points',
		            bbox=bbox, arrowprops=arrowprops)
				ax.annotate('End Point',
		            (ypoints[0],xpoints[0]), xytext=(-2*offset, -offset), textcoords='offset points',
		            bbox=bbox, arrowprops=arrowprops)
		else:
			if(self.endPointRow > self.startPoint[1]):
				ax.annotate('Start Point',
		            (ypoints[0],xpoints[0]), xytext=(-2*offset, -offset), textcoords='offset points',
		            bbox=bbox, arrowprops=arrowprops)
				ax.annotate('End Point',
		            (ypoints[-1],xpoints[-1]), xytext=(offset, offset), textcoords='offset points',
		            bbox=bbox, arrowprops=arrowprops)
			else:
				ax.annotate('Start Point',
		            (ypoints[0],xpoints[0]), xytext=(offset, offset), textcoords='offset points',
		            bbox=bbox, arrowprops=arrowprops)
				ax.annotate('End Point',
		            (ypoints[-1],xpoints[-1]), xytext=(-2*offset, -offset), textcoords='offset points',
		            bbox=bbox, arrowprops=arrowprops)


		if not control_points is None:
			df = pd.read_csv(control_points)
			xpoints = []
			ypoints = []
			cont = 0
			for i in range(1,len(df.index)):
				if cont == 0:
					xpoints.append(df.loc[i][0])
					ypoints.append(df.loc[i][1])
				cont+=1
				if cont == 10:
					cont = 0
			#print(len(xpoints))
			ax.plot(xpoints,ypoints, '-r', label="Sailboat Path")

		self.compute_gps_coordinates((xpoints, ypoints))

		legend = ax.legend(loc='upper center', fontsize='x-large', facecolor='white',framealpha=0.2)

		bbox_wind = dict(boxstyle="circle", fc="white")
		ax.annotate('Wind Direction',
	            (22.3,5.5), xytext=(-2*offset, -offset), textcoords='offset points')
		ax.annotate('              ',
	            (23,4), xytext=(-2*offset, -offset), textcoords='offset points',
	            bbox=bbox_wind)
		bbox_props = dict(boxstyle="rarrow,pad=0.05", fc="white", ec="black", lw=1)
		t = ax.text(20.29, 2.2, "     ", ha="center", va="center", rotation=45,
            size=15,
            bbox=bbox_props)
		
		#ax.annotate('Sp', xy =(xpoints[0],ypoints[0]), xytext = (xpoints[0]-2,ypoints[0]-2),arrowprops=dict(facecolor='white', shrink=0.02))
		
		#ax.yaxis.tick_left()
		
		plt.show()

		
		file_name_png = self.test_name + ".png"
		file_name_pdf = self.test_name + ".pdf"
		fig.savefig(file_name_png, bbox_inches='tight')
		fig.savefig(file_name_pdf, bbox_inches='tight')
		


	def get_current_state(self, point):
		(x, y) = point
		return x*100 + y*10 + x

	def train(self, learning_rate, discount, random_explore, horizon, blocks):
		qtable = self.define_action_reward()
		only_diagonal=False
		rewards, distance = self.generate_reward_distance_map(self.endPoint, only_diagonal)
		self.setDistace(distance)
		self.setRewards(rewards)

		if(self.is_blocked(blocks, self.startPoint) or self.is_blocked(blocks, self.endPoint)):
			print("Please, verify the start point position and end point position! They are placed on blocked square!")
			exit(0)

		state_1_plot = []
		state_2_plot = []
		state_3_plot = []
		state_4_plot = []
		state_5_plot = []

		dict_plot = {}
		summ = 0
		cont = 0
		old_value_1 = 0
		old_value_2 = 0
		old_value_3 = 0
		correct = 0
		wrong = 0
		for i in range(horizon):
			
			#print("Step: {}".format(i), end='\r')
			nboatrl = TrajectoryGenerationNBoat(self.startPoint, self.endPoint, self.matrix_size, self.test_name)

			nboatrl.setDistace(distance)
			nboatrl.setRewards(rewards)
			end_step = False
			while not end_step:
				
				#current_state = nboatrl.positionRow*100+ (nboatrl.positionCol*10)+nboatrl.positionRow
				current_state = self.get_current_state((nboatrl.positionRow, nboatrl.positionCol))
				max_reward_action = qtable[current_state].idxmax()
				#print(current_state)

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
				#if(current_state == self.get_current_state((15,15))):

				cont = cont + 1
				if(reward < 0):
					wrong = wrong + 1
				else:
					correct = correct + 1
			if i > 1:
				if cont > 0:
					state_1_plot.append((wrong/cont)*100)
					print("Error: {}".format(wrong/cont), end='\r')
				#if(current_state == self.get_current_state((15,15))):
			#		max_reward_action = qtable[current_state].idxmax()
				#	old_value_1 = qtable.loc[max_reward_action, current_state] - 100
				#	state_1_plot.append(qtable.loc[max_reward_action, current_state] - 100)
				#else:
				#	state_1_plot.append(old_value_1)

				#if(current_state == self.get_current_state((18,18))):
				#	max_reward_action = qtable[current_state].idxmax()
				#	old_value_2 = qtable.loc[max_reward_action, current_state] - 100
				#	state_2_plot.append(qtable.loc[max_reward_action, current_state] - 100)
				#else:
				#	state_2_plot.append(old_value_2)

				#if(current_state == self.get_current_state((9,9))):
				#	max_reward_action = qtable[current_state].idxmax()
				#	old_value_3 = qtable.loc[max_reward_action, current_state] - 100
				#	state_3_plot.append(qtable.loc[max_reward_action, current_state] - 100)
				#else:
				#	state_3_plot.append(old_value_3)

			#if cont == 0:
			#	x_plot.append(0)
			#else:
			#	x_plot.append(summ/cont)
			#summ = 0
			#cont = 0
										#print(max_reward_action, current_state, (1 - learning_rate) * qtable[current_state][max_reward_action] + learning_rate * learned_value)
			#print("------------------")

		plt.plot(state_1_plot, '-b', label="Training Error", linewidth=2)
		plt.title("Training Error")
		plt.xlabel("Episodes")
		plt.ylabel("%")
		#plt.plot(state_2_plot, '-g', label="State (18,18)")
		#plt.plot(state_3_plot, '-r', label="State (9,9)")
		plt.legend()
		plt.grid()

		with open(self.test_name + '_training_error.csv', mode='w') as file:
			coordenates_writer = csv.writer(file, delimiter=',')
			coordenates_writer.writerow(['error'])
			coordenates_writer.writerow(state_1_plot)
			print("Error saved in {}".format(self.test_name + 'training_error.csv'))
		#plt.show()

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

		#plt.subplots(figsize=(10,10))
		#plt.plot(xpoints,ypoints, 'or')

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

		with open(self.test_name + '_cartesian_file.csv', mode='w') as file:
			coordenates_writer = csv.writer(file, delimiter=',')
			coordenates_writer.writerow(['x','y'])
			for i in range(0, len(new_x_points)):
				coordenates_writer.writerow([xpoints[i], ypoints[i]])
			print("Coordinates saved in {}".format(self.test_name + '_cartesian_file.csv'))

		with open(self.test_name + '_coordinates_file.csv', mode='w') as file:
			coordenates_writer = csv.writer(file, delimiter=',')
			coordenates_writer.writerow(['latitude','longitude'])
			for i in range(0, len(new_x_points)):
				coordenates_writer.writerow([new_y_points[i], new_x_points[i]])
			print("Coordinates saved in {}".format(self.test_name + '_coordinates_file.csv'))
