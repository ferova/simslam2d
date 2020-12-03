import numpy as np


def create_traj(ttype, h, w, posnum = 1000):

	if ttype == 'lisajous':
		traj = lisajous(h,w,posnum)
	elif ttype == 'squircle':
		traj = squircle(h,w,posnum)
	elif ttype == 'sin2':
		traj = sin2(h,w,posnum)
	elif ttype == 'layp':
		traj = layp(h,w,posnum)

	traj  = np.append(traj[0][:, np.newaxis], traj[1][:, np.newaxis], axis=1)

	return traj

def lisajous(h, w, posnum):

	t = np.linspace(0, 2*np.pi+0.01, posnum)

	if h>w:
		x = (0.4*np.cos(3*t) + 0.5)*w
		y = (0.4*np.sin(t) + 0.5)*h
	else:
		x = (0.4*np.sin(t) + 0.5)*w
		y = (0.4*np.cos(3*t) + 0.5)*h


	return [x, y]

def squircle(h, w, posnum):

	t = np.linspace(0, 1.95*np.pi-0.1, posnum)

	x = (np.cos(t)*0.4+0.5)*w
	y = (np.sin(t)*0.4+0.5)*h

	return [x, y]

def sin2(h, w, posnum):

	t = np.linspace(0, 3*np.pi, posnum)
	if w>h:
		x = (t / (3*np.pi) * 0.8 + 0.1)*w
		y = (np.sin(t)*np.sin(t) * 0.8 + 0.1)*h
	else:
		x = (np.sin(t)*np.sin(t) * 0.8 + 0.1)*w
		y = (t / (3*np.pi) * 0.8 + 0.1)*h		

	return [x, y]

def layp(h, w, posnum):

	t = np.linspace(0.95, 4.7, posnum)

	x = (np.cos(t) * 0.5 + 0.6)*w
	y = (np.sin(2*t) * 0.4 + 0.5)*h

	return [x, y]

import cv2


def get_overlap_ratio(rect1, rect2, height, width):

	from shapely.geometry import Polygon

	centre1, theta1 = rect1[:2], rect1[2]
	centre2, theta2 = rect2[:2], rect2[2]

	c, s = np.cos(theta1), np.sin(theta1)
	R1 = np.matrix('{} {}; {} {}'.format(c, -s, s, c))

	c, s = np.cos(theta2), np.sin(theta2)
	R2 = np.matrix('{} {}; {} {}'.format(c, -s, s, c))


	p1 = [ + width / 2,  + height / 2]
	p2 = [- width / 2,  + height / 2]
	p3 = [ - width / 2, - height / 2]
	p4 = [ + width / 2,  - height / 2]

	p1_new = np.dot(p1, R1)+ centre1
	p2_new = np.dot(p2, R1)+ centre1
	p3_new = np.dot(p3, R1)+ centre1
	p4_new = np.dot(p4, R1)+ centre1
	rotatedrect1 = Polygon([p1_new.tolist()[0], p2_new.tolist()[0], p3_new.tolist()[0], p4_new.tolist()[0]])

	p1 = [ + width / 2,  + height / 2]
	p2 = [- width / 2,  + height / 2]
	p3 = [ - width / 2, - height / 2]
	p4 = [ + width / 2,  - height / 2]

	p1_new = np.dot(p1, R2)+ centre2
	p2_new = np.dot(p2, R2)+ centre2
	p3_new = np.dot(p3, R2)+ centre2
	p4_new = np.dot(p4, R2)+ centre2

	rotatedrect2 = Polygon([p1_new.tolist()[0], p2_new.tolist()[0], p3_new.tolist()[0], p4_new.tolist()[0]])

	ratio = rotatedrect1.intersection(rotatedrect2).area / rotatedrect1.union(rotatedrect2).area

	return ratio