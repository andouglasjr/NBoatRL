import numpy as np 
import pandas as pd 
import random
import matplotlib.pyplot as plt
import math

class NBoatRL:
	rewards = None
	positionCol = None
	positionRow = None


	def __init__(self,  distance, rewards, self.positionCol=1, self.positionRow=1, finalCol=5, finalRow = 5, nrow = 5, ncol = 5):
		self.distance = distance
		self.rewards = rewards
		self.positionCol = self.positionCol
		self.positionRow = self.positionRow
		self.finalCol = finalCol
		self.finalRow = finalRow
		self.self.positionRow = self.positionRow
		self.self.positionCol = self.positionCol
		self.nrow = nrow
		self.ncol = ncol

	def move(self, direction, blocks):
		reward = 0
		end = False
		distance_before = self.distance[self.positionCol][self.positionRow]

		if direction == 'Up':
			self.positionRow -=1
		elif direction=='Down':
			self.positionRow +=1
		elif direction=='Left':
			self.positionCol -= 1
		elif direction=='Right':
			self.positionCol += 1
		elif direction=='UpLeft':
			self.positionRow -=1
			self.positionCol -= 1
		elif direction=='UpRight':
			self.positionRow -=1
			self.positionCol += 1
		elif direction == 'DownLeft':
			self.positionRow +=1
			self.positionCol -= 1
		elif direction == 'DownRight':
			self.positionRow +=1
			self.positionCol += 1

		#adding blocks
		#if not blocks == None:
		if (is_blocked(blocks, (self.positionRow, self.positionCol))):
			#print(blocks, self.positionRow, self.positionCol)
			reward = -1000
			end = True
			#print("It is a block!")

		#check if we lost
		elif self.positionRow < 1 or self.positionRow > self.nrow or self.positionCol < 1 or self.positionCol > self.ncol:
			end = True
			reward = -1000
			#print("Ops! Value out of the table range!")
			

		#check if we have reached the end
		elif self.positionCol == self.finalCol and self.positionRow == self.finalRow:
			end = True
			reward = self.rewards[self.positionCol][self.positionRow].item()
			#print("End!")

		else:
			#print("Changing qTable values!")
			end = False
			if distance_before.item() < self.distance[self.positionCol][self.positionRow].item():
				reward = -1000
			else:
				reward = self.rewards[self.positionCol][self.positionRow].item()

		return (reward, end)

	


	def create_map(self, qtable, nrow, ncol, blocks=None):
		data_index_x = []
		for i in range(0,nrow):
			data_index_x.append(i+1)
		map = pd.DataFrame(0, index=data_index_x, columns=data_index_x)
#adding blocks
		self.positionRow = self.self.positionRow
		self.positionCol = self.self.positionCol
		end_step = False

		while not end_step:
			current_state_ = self.positionRow*100+ self.positionCol*10 + self.positionRow
			#if random.random() < 0.1:
			#	direction_ = qtable[current_state_].nlargest(2).index.all()
			#else:
			direction_ = qtable[current_state_].idxmax()

			#print(self.positionRow)
			
			#print("Direction: {}".format(direction_))
			map.loc[self.positionRow, self.positionCol] = 12
			#print(self.positionRow, self.positionCol)
			#print("Direction: {}".format(direction_))
			if direction_=='Up':
				self.positionRow -= 1
			elif direction_=='Down':
				self.positionRow += 1
			elif direction_ == 'Left':
				self.positionCol -= 1
			elif direction_=='Right':
				self.positionCol += 1
			elif direction_=='UpLeft':
				self.positionRow -=1
				self.positionCol -= 1
			elif direction_=='UpRight':
				self.positionRow -=1
				self.positionCol += 1
			elif direction_ == 'DownLeft':
				self.positionRow +=1
				self.positionCol -= 1
			elif direction_ == 'DownRight':
				self.positionRow +=1
				self.positionCol += 1

			if self.positionCol == self.finalCol and self.positionRow == self.finalRow:
				end_step = True
				map.loc[self.positionRow, self.positionCol] = 12

		map.loc[self.self.positionRow, self.self.positionCol] = 3
		map.loc[self.finalRow, self.finalCol] = 6

		if not blocks == None:
			for b in blocks:
				map.loc[b[0],b[1]] = 5
		return map
		
def there_is(already_generated, point, nrow, ncol):
	for ag in already_generated:
		if point[0] < 1 or point[0] > nrow or point[1] < 1 or point[1] > ncol:
			return True
		if(ag[0] == point[0] and ag[1] == point[1]):
			return True

	return False

def is_blocked(blocks, point):
	for b in blocks:
		if(b[0]==point[0] and b[1]==point[1]):
			return True
	return False

def add_block(blocks, distance):
	for b in blocks:
		distance.loc[b[0],b[1]] = distance.loc[b[0],b[1]] - 1 
	return distance

def generate_reward_distance_map(frow, fcol, nrow, ncol, blocks=None, there_is_wind = True):		
	
	#print(frow, fcol, positionCol, positionRow)
	end_generation = False
	distance = pd.DataFrame(1000, index=np.arange(1,nrow+1), columns=np.arange(1,ncol+1))
	cont = 0
	already_generated = []
	#distance = add_block(blocks, distance)
	if not there_is_wind:
		new_frow = frow*math.sqrt(3)/2-fcol*(1/2)
		new_fcol = frow*1/2 + fcol*math.sqrt(3)/2
		frow = round(new_frow,0)
		fcol = round(new_fcol,0)
		#print(round(frow,0), round(fcol,0))
	positionCol, positionRow = fcol, frow
	while not end_generation:
		if (positionRow == frow and positionCol == fcol):
			distance.loc[positionRow,positionCol] = cont
			positionRow = 1000
			already_generated.append((frow, fcol))
			#print(already_generated)
		cont = cont+1
		
		for ag in already_generated:
			(i, j) = ag
			if(not there_is(already_generated,(i+1,j), nrow, ncol)):
				distance.loc[i+1,j] = distance.loc[i,j] + 1
				already_generated.append((i+1, j))
			if(not there_is(already_generated, (i,j+1), nrow, ncol)):
				distance.loc[i,j+1] = distance.loc[i,j] + 1
				already_generated.append((i, j+1))
			if(not there_is(already_generated, (i-1,j), nrow, ncol)):
				distance.loc[i-1,j] = distance.loc[i,j] + 1
				already_generated.append((i-1, j))
			if(not there_is(already_generated, (i,j-1), nrow, ncol)):
				distance.loc[i,j-1] = distance.loc[i,j] + 1
				already_generated.append((i, j-1))
			#print(distance)
			
		if len(already_generated) == nrow*ncol:
			end_generation = True
		#print(distance)

	


	rewards = pd.DataFrame(100, index=np.arange(1,nrow+1), columns=np.arange(1,ncol+1))
	max_distance = 0
	for i in range(1,nrow+1):
		for j in range(1, ncol+1):
			if(distance.loc[i,j] > max_distance):
				max_distance = distance.loc[i,j]
	
	for i in range(1,nrow+1):
		for j in range(1, ncol+1):
			rewards.loc[i,j] = abs(distance.loc[i,j] - max_distance)
	
	print(rewards)
	print(distance)
	return rewards, distance

def generate_reward_distance_map_diagonal(frow, fcol, nrow, ncol, blocks=None):		
	
	#print(frow, fcol, positionCol, positionRow)
	end_generation = False
	distance = pd.DataFrame(1000, index=np.arange(1,nrow+1), columns=np.arange(1,ncol+1))
	already_generated = []
	positionCol, positionRow = fcol, frow
	while not end_generation:
		if (positionRow == frow and positionCol == fcol):
			distance.loc[positionRow,positionCol] = 0
			positionRow = 1000
			already_generated.append((frow, fcol))

		
		for ag in already_generated:
			(i, j) = ag
			if(not there_is(already_generated,(i+1,j+1), nrow, ncol)):
				distance.loc[i+1,j+1] = distance.loc[i,j] + 1
				already_generated.append((i+1, j+1))
			if(not there_is(already_generated, (i+1,j-1), nrow, ncol)):
				distance.loc[i+1,j-1] = distance.loc[i,j] + 1
				already_generated.append((i+1, j-1))
			if(not there_is(already_generated, (i-1,j+1), nrow, ncol)):
				distance.loc[i-1,j+1] = distance.loc[i,j] + 1
				already_generated.append((i-1, j+1))
			if(not there_is(already_generated, (i-1,j-1), nrow, ncol)):
				distance.loc[i-1,j-1] = distance.loc[i,j] + 1
				already_generated.append((i-1, j-1))
			if(not there_is(already_generated,(i+1,j), nrow, ncol)):
				distance.loc[i+1,j] = 1000
				already_generated.append((i+1, j))
			if(not there_is(already_generated, (i,j+1), nrow, ncol)):
				distance.loc[i,j+1] = 1000
				already_generated.append((i, j+1))
			if(not there_is(already_generated, (i-1,j), nrow, ncol)):
				distance.loc[i-1,j] = 1000
				already_generated.append((i-1, j))
			if(not there_is(already_generated, (i,j-1), nrow, ncol)):
				distance.loc[i,j-1] = 1000
				already_generated.append((i, j-1))
			#print(distance)
			
		if len(already_generated) == nrow*ncol:
			end_generation = True
		#print(distance)

	rewards = pd.DataFrame(100, index=np.arange(1,nrow+1), columns=np.arange(1,ncol+1))
	max_distance = 0
	for i in range(1,nrow+1):
		for j in range(1, ncol+1):
			if(distance.loc[i,j] > max_distance):
				max_distance = distance.loc[i,j]
	
	for i in range(1,nrow+1):
		for j in range(1, ncol+1):
			rewards.loc[i,j] = abs(distance.loc[i,j] - max_distance)
	
	print(rewards)
	print(distance)
	return rewards, distance

#Parameters
n_row,n_col = 5,5
final_row, final_col = 5,5
start_row, start_col = 1,1
there_is_wind = True

blocks = []
#creatinh blocks points
# for j in range(1, n_row+1):
# 	for i in range(1, n_col+1):
# 		if (j == 1):
# 			blocks.append((j,i))			
# 			#print(blocks)
# 		else:
# 			if (j == 2) and not (i == 9):
# 				blocks.append((j, i))
# 			else:
# 				if j==3 and not i == 3 and not i == 4 and not i == 7 and not i==8 and not i==9:
# 					blocks.append((j, i))
# 				else:
# 					if j==4 and not i == 4 and not i == 5 and not i == 6 and not i==7 and not i==8 and not i==9:
# 						blocks.append((j, i))
# 					else:
# 						if j==5 and not i == 4 and not i == 5 and not i == 6 and not i==7 and not i==8 and not i==9:
# 							blocks.append((j, i))
# 						else:
# 							if j==6 and not i == 4 and not i == 5 and not i == 6 and not i==7 and not i==8:
# 								blocks.append((j, i))
# 							else:
# 								if j==7 and not i == 4 and not i == 5 and not i == 6 and not i==7:
# 									blocks.append((j, i))
# 								else:
# 									if j==8 and not i == 3 and not i == 4:
# 										blocks.append((j, i))	
# 									else:
# 										if j==9 and not i == 3 and not i == 4:
# 											blocks.append((j, i))
# 										else:
# 											if j==10 and not i == 2 and not i == 3:
# 												blocks.append((j, i))
	#print(blocks)

#blocks = [(1,1),(3,3),(3,4),(5,3)]
#blocks.append((6,6))
#blocks.append((5,6))
#print(blocks)

#Wind direction
wind_direction = 45 #degrees:: use 225 to dead zone
print("Wind Direction: {}".format(wind_direction))

#boat_direction
theta = math.atan2((final_row - start_row),(final_col - start_col))
boat_direction = math.degrees(theta)

#boat_direction = (final_row - start_row)^2 
print("Boat direction: {}".format(round(boat_direction,2)))

#verify if boat direction is opposed to wind
interval = wind_direction - boat_direction
#interval = interval + 360 if interval < 0 else interval
print("interval: {}".format(interval))
if  interval >= 150 and interval <= 210:
	print("Dead Zone")
	there_is_wind = False
else:
	print("Out DZ")

#blocks = [(1,2),(2,2),(4,4),(5,4)]
#rewards, distance = generate_reward_distance_map(final_row, final_col, n_row,n_col, blocks)

def generate_columns_qtable(nrow, ncol):
	cols = []
	for i in range(1,nrow+1):
		for j in range(1,ncol+1):
			cols.append(j*100+i*10 + j)

	print(cols)
	return cols

#states are in columns and actions are in rows

learning_rate =1
discount = 0
random_explore=0.1
cols = generate_columns_qtable(n_row, n_col)


if not there_is_wind:
	rewards, distance = generate_reward_distance_map(final_row, final_col, n_row,n_col, blocks)
else:
	rewards, distance = generate_reward_distance_map_diagonal(final_row, final_col, n_row,n_col, blocks)


qtable = pd.DataFrame(100, index=['Up','Down','Left','Right','UpLeft', 'UpRight', 'DownLeft', 'DownRight'], columns=cols)
#qtable = pd.DataFrame(100, index=['Up','Down','Left','Right'], columns=cols)
#qtable = pd.DataFrame(100, index=['UpLeft', 'UpRight', 'DownLeft', 'DownRight'], columns=cols)
print(qtable)

for i in range(1000):
	#print ("Game # " + str(i))
	nboatrl = NBoatRL(distance=distance, rewards=rewards, self.positionCol=start_col, self.positionRow=start_row, finalCol = final_col,  finalRow = final_row, nrow = n_row, ncol=n_col )
	end_step = False
	while not end_step:
		current_state = nboatrl.positionRow*100+ (nboatrl.positionCol*10)+nboatrl.positionRow
		#print(current_state)
		max_reward_action = qtable[current_state].idxmax()
		#print("Max Reward: {}".format(max_reward_action))

		if random.random() < random_explore:
			max_reward_action = qtable.index[random.randint(0,len(qtable.index)-1)]
			#max_reward_action = qtable[current_state].nlargest(random.randrange(4)).index.all()
			#print(max_reward_action)

		reward, end_step = nboatrl.move(max_reward_action, blocks)

		if end_step:
			qtable.loc[max_reward_action, current_state] = reward

		else:
			#print("CS:" + str(current_state) + ", Action: " + max_reward_action + ", Reward: " + str(reward))
			#print(qtable.columns)
			#print("nboatrl.positionCol: {}, nboatrl.positionRow: {}".format(nboatrl.positionCol, nboatrl.positionRow))
			opimtal_future_value = qtable[nboatrl.positionRow*100 + (nboatrl.positionCol*10)+nboatrl.positionRow].max()
			discounted_opimtal_future_value = discount * opimtal_future_value
			learned_value = reward + discounted_opimtal_future_value
			qtable.loc[max_reward_action, current_state] = (1 - learning_rate) * qtable[current_state][max_reward_action] + learning_rate * learned_value

		

print(qtable)
#print(qtable[22],qtable[23],qtable[24],qtable[25],qtable[26])
map_ = nboatrl.create_map(qtable, n_row, n_col, blocks)
print(map_)

fig, ax = plt.subplots()
#min_value, max_value = 0, 200
ax.matshow(map_, cmap=plt.cm.Paired)
#plt.matshow(map_)
#plt.show()
xpoints = []
ypoints = []
points = []
for i in range(1,n_row+1):
	for j in range(1,n_col+1):
		if(map_.loc[i,j]==12) or (map_.loc[i,j] == 3) or (map_.loc[i,j]==6):
			xpoints.append(j-1)
			ypoints.append(i-1)
print(xpoints, ypoints)
#axes = plt.gca()
ax.set_xlim([-1.5,n_row])
ax.set_ylim([-1.5,n_col])
plt.grid(b=True, axis='both')

major_ticks = np.arange(-1.5, n_row+1, 1)
ax.set_xticks(major_ticks)
major_ticks_y = np.arange(-1.5, n_row+1, 1)
ax.set_yticks(major_ticks_y)
ax.grid(which='both')


#plt.xticks(np.arange(0.5,n_row+0.5, 1))
#plt.xticks(np.arange(1,n_col, 1))
ax.plot(xpoints,ypoints, 'o')
#U,V = np.meshgrid(xpoints, ypoints)
#for i in range(0, len(xpoints)-1):
#	ax.arrow(xpoints[i], ypoints[i], abs(xpoints[i+1] - xpoints[i]), abs(ypoints[i+1] - ypoints[i]))
ax.annotate('WD', xy =(2,12), xytext = (2,15),arrowprops=dict(facecolor='black', shrink=0.05))


plt.show()
#map_.plot.scatter(x='index', y='columns')
#plt.show()