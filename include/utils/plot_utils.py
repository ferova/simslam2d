import numpy as np
import cv2
import matplotlib.pyplot as plt

def draw_rectangle(image, centre, theta, width, height, linewidth = 3, color = (255, 0, 0)):

	c, s = np.cos(theta), np.sin(theta)
	R = np.matrix('{} {}; {} {}'.format(c, -s, s, c))


	p1 = [ + width / 2,  + height / 2]
	p2 = [- width / 2,  + height / 2]
	p3 = [ - width / 2, - height / 2]
	p4 = [ + width / 2,  - height / 2]
	p5 = [0, - height / 2]
	p1_new = np.dot(p1, R)+ centre
	p2_new = np.dot(p2, R)+ centre
	p3_new = np.dot(p3, R)+ centre
	p4_new = np.dot(p4, R)+ centre
	p5_new = np.dot(p5, R)+ centre
	centre =np.asarray(centre)[np.newaxis, :]

	img = cv2.line(image, (int(p1_new[0, 0]), int(p1_new[0, 1])), (int(p2_new[0, 0]), int(p2_new[0, 1])), color, linewidth)
	img = cv2.line(img, (int(p2_new[0, 0]), int(p2_new[0, 1])), (int(p3_new[0, 0]), int(p3_new[0, 1])), color, linewidth)
	img = cv2.line(img, (int(p3_new[0, 0]), int(p3_new[0, 1])), (int(p4_new[0, 0]), int(p4_new[0, 1])), color, linewidth)
	img = cv2.line(img, (int(p4_new[0, 0]), int(p4_new[0, 1])), (int(p1_new[0, 0]), int(p1_new[0, 1])), color, linewidth)
	img = cv2.line(img, (int(p2_new[0, 0]), int(p2_new[0, 1])), (int(p4_new[0, 0]), int(p4_new[0, 1])), color, linewidth)
	img = cv2.line(img, (int(p1_new[0, 0]), int(p1_new[0, 1])), (int(p3_new[0, 0]), int(p3_new[0, 1])), color, linewidth)
	img = cv2.line(img, (int(centre[0, 0]), int(centre[0, 1])), (int(p5_new[0, 0]), int(p5_new[0, 1])), color, linewidth)

	return img

def plot_rectangles(map_image2, rect1, rect2, h, w, show = True):




	map_image = map_image2.copy()

	map_image = draw_rectangle(map_image, rect1[:2], rect1[-1], w, h, color = (255, 0, 0))

	if len(rect2) == 3:

		x2, y2 = rect2[:2]

		theta2 = rect2[-1]

		map_image = draw_rectangle(map_image,  [x2, y2], theta2, w, h, color = (0, 0, 255))


	if show:

		cv2.namedWindow('window1' , cv2.WINDOW_NORMAL)
		cv2.imshow('window1', map_image)
		cv2.waitKey(1)
	else:
		return map_image

def plot_graph(Map):

	nodes, edges = Map[-2:]
	x = [item[0] for item in nodes]
	y = [-item[1] for item in nodes]
	# plt.figure()
	plt.scatter(x, y, c='k', s=100)

	for i in range(len(edges)):
		#print(edges[i])
		id1 = int(edges[i][0][0])
		id2 = int(edges[i][0][1])
		plt.plot([nodes[id1-1][0], nodes[id2-1][ 0]], [-nodes[id1-1][1], -nodes[id2-1][ 1]], linewidth = 1, color = 'k')

	plt.show()

def reconstruct_mapview(Map, border_size = 500, draw_rectangles = False, live_plot = False):

	features, keyframes, nodes, edges = Map

	x =[x[0] for x in nodes]
	y =[y[1] for y in nodes]
	theta =[z[2] for z in nodes]


	x_min = np.amin(np.asarray(x))
	x_max = np.amax(np.asarray(x))
	y_min = np.amin(np.asarray(y))
	y_max = np.amax(np.asarray(y))

	h = y_max - y_min
	w = x_max - x_min

	x = x-x_min
	y = y-y_min

	x_ini = x[0]
	y_ini = y[0]
	theta_ini = theta[0]


	Ti = np.eye(3)

	Ti[0,2] = x_ini+border_size
	Ti[1,2] = y_ini+border_size

	im0 = keyframes[0][0]
	rows, cols, _ = im0.shape 


	Ri      = np.eye(3)
	Ri[0:2, :]= cv2.getRotationMatrix2D((cols/2, rows/2), np.degrees(theta_ini),1)
	Ri[0,2] = -(cols/2)*np.cos(theta_ini)-np.sin(theta_ini)*rows/2
	Ri[1,2] = (-(cols/2)*np.sin(theta_ini)+np.cos(theta_ini)*rows/2)*-1

	a = h+border_size*2 	
	b = w+border_size*2

	canvas = cv2.warpPerspective(im0, np.matmul(Ti,Ri), (int(b), int(a)), borderValue=(255, 255, 255))

	if draw_rectangles:
		canvas = plot_rectangles(canvas, [x_ini+border_size, y_ini+border_size, theta_ini], [], rows, cols, show = False)

	for i in range(len(nodes)-1):

		im1 = keyframes[i+1][0]
		rows, cols, _ = im1.shape 	
		xi, yi, thetai = x[i+1], y[i+1], theta[i+1]

		Ti      = np.eye(3)
		Ti[0,2] = xi+border_size
		Ti[1,2] = yi+border_size


		Ri      = np.eye(3)
		Ri[0:2, :]= cv2.getRotationMatrix2D((cols/2,rows/2), np.degrees(thetai),1)
		Ri[0,2] = -(cols/2)*np.cos(thetai)-np.sin(thetai)*rows/2
		Ri[1,2] = (-(cols/2)*np.sin(thetai)+np.cos(thetai)*rows/2)*-1

		canvas = cv2.warpPerspective(im1, np.matmul(Ti,Ri), (int(b), int(a)), canvas.copy(), borderMode=cv2.BORDER_TRANSPARENT)

		if draw_rectangles:
			canvas = plot_rectangles(canvas, [xi+border_size, yi+border_size, thetai], [], rows, cols, show = False)
		if live_plot:
			cv2.namedWindow('window1' , cv2.WINDOW_NORMAL)
			cv2.imshow('window1', canvas)
			cv2.waitKey(1)

	cv2.namedWindow('window1' , cv2.WINDOW_NORMAL)
	cv2.imshow('window1', canvas)
	cv2.waitKey(0)

def reconstruct_trajectory(Map, gp_size, border_size = 0, draw_rectangles = False, live_plot = False):

	features, keyframes, nodes, edges = Map

	x =[x[0] for x in nodes]
	y =[y[1] for y in nodes]
	theta =[z[2] for z in nodes]


	x_min = np.amin(np.asarray(x))
	x_max = np.amax(np.asarray(x))
	y_min = np.amin(np.asarray(y))
	y_max = np.amax(np.asarray(y))

	h = gp_size[0]
	w = gp_size[1]

	#x = x-x_min
	#y = y-y_min

	x_ini = x[0]
	y_ini = y[0]
	theta_ini = theta[0]


	Ti = np.eye(3)

	Ti[0,2] = x_ini+border_size
	Ti[1,2] = y_ini+border_size

	im0 = keyframes[0][0]
	rows, cols, _ = im0.shape 


	Ri      = np.eye(3)
	Ri[0:2, :]= cv2.getRotationMatrix2D((cols/2, rows/2), np.degrees(theta_ini),1)
	Ri[0,2] = -(cols/2)*np.cos(theta_ini)-np.sin(theta_ini)*rows/2
	Ri[1,2] = (-(cols/2)*np.sin(theta_ini)+np.cos(theta_ini)*rows/2)*-1

	a = h+border_size*2 	
	b = w+border_size*2

	canvas = cv2.warpPerspective(im0, np.matmul(Ti,Ri), (int(b), int(a)), borderValue=(255, 255, 255))

	if draw_rectangles:
		canvas = plot_rectangles(canvas, [x_ini+border_size, y_ini+border_size, theta_ini], [], rows, cols, show = False)

	for i in range(len(nodes)-1):

		im1 = keyframes[i+1][0]
		rows, cols, _ = im1.shape 	
		xi, yi, thetai = x[i+1], y[i+1], theta[i+1]

		Ti      = np.eye(3)
		Ti[0,2] = xi+border_size
		Ti[1,2] = yi+border_size


		Ri      = np.eye(3)
		Ri[0:2, :]= cv2.getRotationMatrix2D((cols/2,rows/2), np.degrees(thetai),1)
		Ri[0,2] = -(cols/2)*np.cos(thetai)-np.sin(thetai)*rows/2
		Ri[1,2] = (-(cols/2)*np.sin(thetai)+np.cos(thetai)*rows/2)*-1

		canvas = cv2.warpPerspective(im1, np.matmul(Ti,Ri), (int(b), int(a)), canvas.copy(), borderMode=cv2.BORDER_TRANSPARENT)

		if draw_rectangles:
			canvas = plot_rectangles(canvas, [xi+border_size, yi+border_size, thetai], [], rows, cols, show = False)
		if live_plot:
			cv2.namedWindow('window1' , cv2.WINDOW_NORMAL)
			cv2.imshow('window1', canvas)
			cv2.waitKey(1)

	cv2.namedWindow('window1' , cv2.WINDOW_NORMAL)
	cv2.imshow('window1', canvas)
	cv2.waitKey(0)
	return canvas

