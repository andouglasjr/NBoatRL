import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('map.jpg',cv2.IMREAD_COLOR)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#cv2.imwrite('new_image.jpg', gray)
new_image = []
width, height, channel = img.shape
print(img.shape)
n = 15
square_size = round(width/n)
point_position = round(square_size - square_size/2)
print(point_position)

col, row = 0,0

matrix = []
for i in range(width - 14, 14, -square_size):
	matrix_row = []
	for j in range(14,height, square_size):
		gray_value = gray[i][j]
			
		if (gray_value > 65 and gray_value < 80):
			matrix_row.append((row,col))
		row += 1
	col += 1
	row = 0
	matrix.append(matrix_row)
#compute crop
#print(matrix)
xpoints = []
ypoints = []
for row in matrix:
	for i in row:
		xpoints.append(i[0])
		ypoints.append(i[1])
plt.plot(xpoints,ypoints, 'o')

plt.show()
image_croped = gray
image_croped = np.array(image_croped)
#cv2.imwrite('new_image.jpg', image_croped)