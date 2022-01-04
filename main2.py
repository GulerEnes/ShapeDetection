import cv2 as cv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
import math

sys.setrecursionlimit(10 ** 7)


def is_point_edge(src, row, col):
	sum_ = int(-src[row][col])
	for i in range(row - 1, row + 2):
		for j in range(col - 1, col + 2):
			sum_ += int(src[i][j])
	return False if sum_ == 0 else True


def edge_points(src):
	row, col = src.shape
	edges = np.zeros(src.shape, dtype=np.uint8)

	for i in range(1, row - 1):
		for j in range(1, col - 1):
			if src[i][j] == 0 and is_point_edge(src, i, j):
				edges[i][j] = 255
	return edges


def neighboors(row, col):
	n = list()
	for i in range(row - 1, row + 2):
		for j in range(col - 1, col + 2):
			n.append((i, j))
	return n


def DFS(src, visits, row, col, cluster):
	visits[row][col] = 255
	n = neighboors(row, col)
	for i, j in n:
		if src[i][j] == 255 and visits[i][j] == 0:
			cluster.add((i, j))
			DFS(src, visits, i, j, cluster)


def cluster_edges(src):
	visits = np.zeros(src.shape, dtype=np.uint8)
	clusters = list()

	row, col = src.shape
	for i in range(1, row - 1):
		for j in range(1, col - 1):
			if src[i][j] == 255 and visits[i][j] == 0:
				cluster = set()
				DFS(src, visits, i, j, cluster)
				clusters.append(cluster)
	return clusters


def bounding_box_of_shape(cluster):
	points = list(cluster)
	points.sort(key=lambda x: x[0])
	min_x, max_x = points[0][0], points[-1][0]
	points.sort(key=lambda x: x[1])
	min_y, max_y = points[0][1], points[-1][1]

	return min_y, min_x, max_y - min_y, max_x - min_x,


def test(src, points):
	temp = np.zeros(src.shape, dtype=np.uint8)
	for i, j in points:
		temp[i][j] = 255
	cv.imshow("temp", temp)
	cv.waitKey()


def length_between_two_points(p1, p2):
	"""
	This function calculates length between two point
	:param p1: a point as (x, y)
	:param p2: a point as (y, x)
	:return: length in pixel
	"""
	x1, y1 = p1
	y2, x2 = p2
	return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def detect_field(center, point):
	"""
	This function detects the field of point according to origin. Assuming center point is origin.
	:param center: Origin point. Format: (x, y)
	:param point: Target point to detect field. Format: (x, y)
	:return: Integer value of field. Can ony be 1, 2, 3 or 4.

	#      |
	#   2  |  1
	# -----C-----
	#   3  |  4
	#      |
	"""

	cx, cy = center
	py, px = point

	if px > cx:
		return 1 if py > cy else 4
	return 2 if py > cy else 3


def angle_of_a_line(p1, p2):
	x1, y1 = p1
	y2, x2 = p2

	a = abs(y2 - y1)
	b = abs(x2 - x1)
	if b == 0:
		b = 0.0000001

	alpha = math.degrees(math.atan(a / b))
	field = detect_field(p1, p2)

	if field == 1:
		return alpha
	elif field == 2:
		return 180 - alpha
	elif field == 4:
		return 180 + alpha
	elif field == 3:
		return 360 - alpha


image = cv.imread("shapes6.jpeg", 0)
blured = cv.GaussianBlur(image, (7, 7), 0)
_, th = cv.threshold(blured, 180, 255, cv.THRESH_BINARY)

edges = edge_points(th)

clusters = cluster_edges(edges)
print("Number of found shapes:", len(clusters))

for shape in clusters:
	# test(image,shape)
	x, y, w, h = bounding_box_of_shape(shape)

	center_point = (x + w // 2, y + h // 2)
	# cv.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 5)
	# cv.imshow("image", image)
	# cv.waitKey()

	points = list(shape)
	length_axis = [length_between_two_points(center_point, point) for point in points]
	angle_axix = [angle_of_a_line(center_point, point) for point in points]
	l = list(zip(angle_axix, length_axis))

	graph = sorted(l, key=lambda x: x[0])
	# print(graph)
	# angle_axix.sort()
	# cv.circle(image, center_point, 6, (255, 0, 0), thickness=-1)
	print(center_point, points[0])
	# cv.circle(image, (points[0][1], points[0][0]), 6, (255, 0, 0), thickness=-1)
	# cv.imshow("image", image)
	# cv.waitKey()

	angle_axix = [i for i, j in graph]
	length_axis = [j for i, j in graph]

	print("min_l:", int(min(length_axis)), "   max_l:", int(max(length_axis)), "   dif:", int(max(length_axis)-min(length_axis)))
	print("---------------------------------------------------------------------------")
	plt.plot(angle_axix, length_axis)

	plt.ylabel('length_axis')
	plt.xlabel('angle_axix')
	plt.title('shape_graph')
	plt.show()

# cv.imshow("th", th)
# cv.imshow("edges", edges)
cv.waitKey()
