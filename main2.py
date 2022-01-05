import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import sys
import math

sys.setrecursionlimit(10 ** 7)
shape_names = {0: 'circle',
			   3: 'triangle',
			   5: 'pentagon',
			   6: 'hexagon'}


def is_point_edge(src, row, col):
	"""
	This function checks a specific point to detect is an edge point or not.
	If the point is black and one of the neighbors of that point is white, then the point is an edge point
	:param src: Source image
	:param row: row index
	:param col: column index
	:return: Boolean
	"""
	for i in range(row - 1, row + 2):
		for j in range(col - 1, col + 2):
			if src[i][j] == 255:
				return True
	return False


def edge_points(src):
	"""
	This function creates a 3x3 window and slides it on the whole image.
	If a point is an edge point. Then mark it as white color
	:param src: Source image
	:return: Marked edges. Edge points are white, others black
	"""
	row, col = src.shape
	edges = np.zeros(src.shape, dtype=np.uint8)

	for i in range(1, row - 1):
		for j in range(1, col - 1):
			if src[i][j] == 0 and is_point_edge(src, i, j):
				edges[i][j] = 255
	return edges


def neighbors(row, col):
	"""
	This function finds all 8 neighbors of a point and points itself.
	points themselves doesn't affect the algorithm because it is already black
	:param row: row index
	:param col: column index
	:return: neighbors list
	"""
	n = list()
	for i in range(row - 1, row + 2):
		for j in range(col - 1, col + 2):
			n.append((i, j))
	return n


def DFS(src, visits, row, col, cluster):
	visits[row][col] = 255
	n = neighbors(row, col)
	for i, j in n:
		if src[i][j] == 255 and visits[i][j] == 0:
			cluster.add((i, j))
			DFS(src, visits, i, j, cluster)


def cluster_edges(src):
	"""
	This function finds each shape's points. Points are stored in sets. And sets are stored in a list.
	To be able to cluster points for each shape separately DFS algorithm was used.
	:param src: Source image
	:return: clustered edge points of each shape as a list
	"""
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
	"""
	This function finds top left corner coordinate, with, and height of bounding box of given set of points
	:param cluster: set of points
	:return: top_left_x, top_left_y, width, height of box
	"""
	points = list(cluster)
	points.sort(key=lambda x: x[0])
	min_x, max_x = points[0][0], points[-1][0]
	points.sort(key=lambda x: x[1])
	min_y, max_y = points[0][1], points[-1][1]

	return min_y, min_x, max_y - min_y, max_x - min_x,


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
	:param point: Target point to detect field. Format: (y, x)
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
	"""
	This function calculates the angle of the line. The line is calculated with two given points
	:param p1: point1 Format: (x, y)
	:param p2: point2 Format: (y, x)
	:return:
	"""
	x1, y1 = p1
	y2, x2 = p2

	a = abs(y2 - y1)
	b = abs(x2 - x1)
	if b == 0:  # tan(90) is 0. So, to avoid zero division error we adding very small number
		b = 0.0000001
	alpha = math.degrees(math.atan(a / b))

	# We want the angle in range 0 to 360. To be able to calculate in this range
	# we used detect_field() function and then decided different calculations for each field.
	field = detect_field(p1, p2)
	if field == 1:
		return alpha
	elif field == 2:
		return 180 - alpha
	elif field == 4:
		return 180 + alpha
	elif field == 3:
		return 360 - alpha


def shift_graph(graph):
	"""
	This function finds min point of y axis. And left side of that point is shifting to right side of graph.
	:param graph: list of points. Format (angle, length)
	:return: None
	"""
	# Find min length
	min_l = 100000
	min_l_ind = -1
	for ind, pair in enumerate(graph):
		a, l = pair
		if l < min_l:
			min_l = l
			min_l_ind = ind

	# Shifting
	while min_l_ind > -1:
		pair = graph.pop(0)
		a, l = pair
		graph.append((a + 360, l))

		min_l_ind -= 1


def smooth_graph(graph):
	"""
	This function smoothes the graph by finding all peak points.
	:param graph: list of points. Format (angle, length)
	:return: graph of peak points
	"""
	peaks_graph = []
	i = 1
	while i < len(graph) - 1:
		a, l = graph[i]
		_, l_left = graph[i - 1]
		_, l_right = graph[i + 1]

		if l > l_left and l > l_right:  # It is a peak
			peaks_graph.append((a, l))
		i += 1
	return peaks_graph


def detect_corners(graph):
	"""
	This function finds peak points and filtering with some values to find and count corner points
	:param graph: list of points. Format (angle, length)
	:return: number of corners, corner points as a list
	"""
	lengths = [l for _, l in graph]

	min_l = min(lengths)
	max_l = max(lengths)
	diff = max_l - min_l
	length_limit = min_l + diff * 0.45  # A corner should be over on this limit
	angle_limit = 40  # between two corner there must be at least 40 degree
	if diff < 15:  # There is no corner. So, it is a circle
		return 0, []

	corners = list()
	corner_count = 0
	i = 1
	last_peak_angle = -1000
	while i < len(graph) - 1:
		a, l = graph[i]
		_, l_left = graph[i - 1]
		_, l_right = graph[i + 1]

		if l > l_left and l > l_right:  # It is a peak
			if l > length_limit and a - last_peak_angle > angle_limit:
				corner_count += 1
				last_peak_angle = a
				corners.append((a, l))
		i += 1
	return corner_count, corners


def show(window_name, img):
	"""
	This is a short way to show images in a window
	:param window_name: window title name
	:param img: source image
	:return: None
	"""
	cv.imshow(window_name, img)
	cv.waitKey()


image = cv.imread("shapes6.jpeg")
# show("image", image)

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

blured = cv.GaussianBlur(gray, (21, 21), 0)
# show("blured", blured)

_, th = cv.threshold(blured, 180, 255, cv.THRESH_BINARY)
# show("th", th)

edges = edge_points(th)
# show("edges", edges)

clusters = cluster_edges(edges)
print("Number of found shapes:", len(clusters))

for shape in clusters:
	# print("---------------------------------------------------------------------------")
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
	graph = list(sorted(l, key=lambda x: x[0]))

	# print(graph)
	# angle_axix.sort()
	# cv.circle(image, center_point, 6, (255, 0, 0), thickness=-1)
	# print(center_point, points[0])
	# cv.circle(image, (points[0][1], points[0][0]), 6, (255, 0, 0), thickness=-1)
	# cv.imshow("image", image)
	# cv.waitKey()
	# angle_axix = [i for i, j in graph]
	# length_axis = [j for i, j in graph]
	# print("min_l:", int(min(length_axis)), "   max_l:", int(max(length_axis)), "   dif:",
	# 	  int(max(length_axis) - min(length_axis)))

	# plt.plot(angle_axix, length_axis)
	#
	# plt.ylabel('length_axis')
	# plt.xlabel('angle_axix')
	# plt.title('shape_graph')
	# plt.show()
	#################################
	shift_graph(graph)

	# angle_axix = [i for i, j in graph]
	# length_axis = [j for i, j in graph]
	#
	# plt.plot(angle_axix, length_axis)
	#
	# plt.ylabel('length_axis')
	# plt.xlabel('angle_axix')
	# plt.title('shape_graph')
	# plt.show()
	#################################
	peaks_graph = smooth_graph(graph)
	#
	# angle_axix = [i for i, j in peaks_graph]
	# length_axis = [j for i, j in peaks_graph]
	#
	# plt.plot(angle_axix, length_axis)
	#
	# plt.ylabel('length_axis')
	# plt.xlabel('angle_axix')
	#################################
	number_of_corners, corners = detect_corners(peaks_graph)
	# print("number_of_corners:", number_of_corners)
	# print("corners:", corners)
	# corner_a = [i for i, j in corners]
	# corner_l = [j for i, j in corners]
	# plt.scatter(corner_a, corner_l)
	# plt.title('num of corners: ' + str(number_of_corners))
	# plt.show()

	font = cv.FONT_HERSHEY_SIMPLEX
	text_position = center_point[0] - 30, center_point[1]
	if number_of_corners == 4:
		# If it has 4 edge it can be square or rectangle
		if abs(w / h - 1) < 0.2:  # For square with/height should be close to 1
			cv.putText(image, 'square', text_position, font, 0.7, (0, 0, 255), 2, cv.LINE_AA)
		else:  # Otherwise it is rectangle
			cv.putText(image, 'rectangle', text_position, font, 0.7, (0, 0, 255), 2, cv.LINE_4)
	else:  # It can be named according to number of edges.
		cv.putText(image, shape_names[number_of_corners], text_position, font, 0.7, (0, 0, 255), 2, cv.LINE_4)

cv.imshow("output", image)
cv.waitKey()
