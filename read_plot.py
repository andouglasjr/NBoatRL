import pandas
import matplotlib.pyplot as plt

angle_dict = {"Right":0, "UpRight":45, "Up":90, "UpLeft":135, "Left":180, "DownLeft":225, "Down":270, "DownRight":315}
boat_direction = 0
wind_direction = 180
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
			
print(angle_dict)