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

	x = (0.4*np.sin(t) + 0.5)*w
	y = (0.4*np.cos(3*t) + 0.5)*h

	return [x, y]

def squircle(h, w, posnum):

	t = np.linspace(0, 2*np.pi-0.1, posnum)

	x = (np.abs(np.cos(t))**(1 / 2)*np.sign(np.cos(t))*0.4 + 0.5)*w
	y = (np.abs(np.sin(t))**(1 / 2)*np.sign(np.sin(t))*0.4 + 0.5)*h

	return [x, y]

def sin2(h, w, posnum):

	t = np.linspace(0, 3*np.pi, posnum)

	x = (t / (3*np.pi) * 0.8 + 0.1)*w
	y = (np.sin(t)*np.sin(t) * 0.8 + 0.1)*h

	return [x, y]

def layp(h, w, posnum):

	t = np.linspace(0.95, 4.7, posnum)

	x = (np.cos(t) * 0.5 + 0.6)*w
	y = (np.sin(2*t) * 0.4 + 0.5)*h

	return [x, y]

