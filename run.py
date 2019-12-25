from TrajectoryGenerationNBoat import TrajectoryGenerationNBoat
import time

start_time = time.time()

#Setup Test
trajectory = 2 #or 2
if trajectory == 1:
	startPoint = (8,8) #8,8
	endPoint = (20,21) #20,21
else:
	startPoint = (20,21) #8,8
	endPoint = (8,8) #20,21
matrix_size = 25
img_path = "map_obstacle"
wind_direction = -45 #310  #200

#Test name
folder_result = "results/"
test_name = folder_result + "traject_"+str(trajectory)+"_"+img_path
#csv_file = 'results/trajectory_2_winddiretion_-45cartesian_file.csv'
csv_file = test_name
#control_file = 'exp2_obstacle_cartesian.csv'
control_file = None
tgBoat = TrajectoryGenerationNBoat(startPoint, endPoint, matrix_size, test_name)
tgBoat.setWindDirection(wind_direction)
blocks = tgBoat.get_map_blocks(img_path+".jpg")
#blocks = []
qTable = tgBoat.train(0.8,0.7,0.3,3000, blocks)
map = tgBoat.create_map(qTable, blocks)
points = tgBoat.show_tracetory_from_file(csv_file, control_file,blocks, map)
#tgBoat.compute_gps_coordinates(points)


#print(points)

#tgBoat.compute_gps_coordinates(points)

time_elapsed = time.time() - start_time
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
