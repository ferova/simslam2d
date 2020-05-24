import numpy as np



def pre_path(path):

	"""
	Pre-process a trajectory given as an array-like [x,y] pair of vectors in order to add an orientation
	at every point. The orientation is caculated such that it is tangent to the trajectory.
	Parameters:
		path - [x,y] list or array that contains the positions of the trajectory.
	Returns:
		path - [x,y,theta] numpy array with angle information included.
	Notes:
		The path is considered to be cyclic so the last angle is found between the last and the first point.
	"""

	path2 = np.roll(path, -1, axis=0)


	dx = path2[:,0]-path[:,0]
	dy = path2[:,1]-path[:,1]

	theta = np.arctan2(dx, dy)+np.pi
	theta[-1] = theta[-2]
	path  = np.append(path, theta[:, np.newaxis], axis = 1)

	return path 




