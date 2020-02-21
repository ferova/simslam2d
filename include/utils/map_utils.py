import numpy as np
import cv2
#from plot_utils import plot_rectangles, plot_graph
#from graph_opt.posegraph import PoseGraph
from PIL import Image
import diagonal_crop
import matplotlib.pyplot as plt
from itertools import compress
from tqdm import tqdm
import os

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 30)
flann = cv2.FlannBasedMatcher(index_params, search_params)

def homography(img1,img2):

	'''
	Finds the homography between an image pair from their feature vectors.
	Parameters:
		img1 - Feature vector of image one as returned by OpenCV detectAndCompute.
		img2 - Feature vector of image two as returned by OpenCV detectAndCompute.
	Returns:
		H - 3x3 numpy array with the corresponding homography.
	Notes:
		When no homography is found returns None.

	'''

	# Values are unpacked.

	kp1, des1 = img1
	kp2, des2 = img2

	# Matcher is created, matches computed.

	matches = flann.knnMatch(des1,des2,k=2)

	# Store all the good matches as per Lowe's ratio test.

	good = []

	for m,n in matches:
		if m.distance < 0.7*n.distance:
			good.append(m)

	src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
	dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

	# Find the homography.

	M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

	# M = cv2.estimateRigidTransform(src_pts, dst_pts, False) # Find the homography as a rigid transform, maybe its more efficient.

	return M

def add2map(im1, im2, h, w, init = None, Map = None, orig = None, theta = 0, save_keyf = False):

	"""
	Adds an image to the map.

	Parameters:
		im1  - Image to be added to the Map.
		h    - Height of the image.
		w    - Width  of the image.
		init - Initialization state of the Map, wether to start at current origin and angle or not.
		orig - Where to begin the map.
		theta- Initial orientation.
	Returns:
		Map - A list that contains the nodes and edges of the graph together with
			  the corresponding keyframes and features of each node (Should keyframes not be stored?).
	Notes:
		If Map is None then creates a new Map with the current 
		image as the first node. If init is None then initializes the current position
		at orig with orientation theta.

	"""

	# Create feature detector.

	surf = cv2.xfeatures2d.SURF_create()

	# Create new map if None is passed.

	if Map == None:
		keyframes = []
		nodes     = []
		edges     = []

	# Initialize Map if None is passed.

	if init == None:

		init = True
		print('init')
		posy, posx = orig
		theta = theta

		# Save keyframes only when needed

		if save_keyf:
			keyframes.append([im1.copy()])
		else:
			keyframes.append([])

		features  = [(surf.detectAndCompute(im1,None))]
		nodes.append(np.array([posx, posy , theta]))   

		Map = [features, keyframes, nodes, edges]
	
	# Add the image to the existing Map otherwise.

	else:

		# Unpack Map and add features of the current image to the map.

		features, keyframes, nodes, edges = Map

		features.append(surf.detectAndCompute(im1,None))

		# Finds the homography between the current and the past image.

		H = homography(features[-1], features[-2])

		# If no homography is found returns the current Map, without adding the last passed image.

		if H is None:
			return Map, init

		# Unpacks the last position.
		
		posx, posy , theta = nodes[-1][0], nodes[-1][1], nodes[-1][2]

		# Compute correction values for rotation around a point different from the origin.

		alpha = H[0, 0]
		beta  = H[0, 1]

		# Compute current angle from homography.

		theta1 = theta

		theta += np.arctan2(H[0, 1], H[0, 0])

		# Compute corrected translation.

		H[0,2] = H[0,2]-((1-alpha)*w/2-beta*h/2)
		H[1,2] = H[1,2]-(beta*w/2+(1-alpha)*h/2)

		# Compute current position from corrected homography.

		posx  += H[0,2]*np.cos(-theta1)-H[1,2]*np.sin(-theta1)
		posy  += H[0,2]*np.sin(-theta1)+H[1,2]*np.cos(-theta1)

		#print(H[0,2]*np.cos(-theta1)-H[1,2]*np.sin(-theta1),H[0,2]*np.sin(-theta1)+H[1,2]*np.cos(-theta1), 'delta', 'deltay')

		# Zero pad image if size is smaller.

		if (im1.shape[0] < h):
			im1 = cv2.copyMakeBorder(im1, top= 0, bottom= h-im1.shape[0],
										 left= 0, right= 0,
										 borderType= cv2.BORDER_CONSTANT, value=[0,0,0] )

		if (im1.shape[1] < w):
			im1 = cv2.copyMakeBorder(im1, top= 0, bottom= 0,
										 left= 0, right= w-im1.shape[1],
										 borderType= cv2.BORDER_CONSTANT, value=[0,0,0] )

		# Append everything to existing map and repack it.

		if save_keyf:
			keyframes.append([im1.copy()])
		else:
			keyframes.append([])

		nodes.append(np.array([posx, posy , theta]))    
		edges.append([np.hstack((len(nodes)-1,np.hstack((len(nodes), np.array(H).flatten())))).T])

		Map = [features, keyframes, nodes, edges]

	return Map, init

def optimize_map(Map):

	"""
	Optimizes the pose-graph associated with the given Map.

	Parameters:
		Map - Map to opmitize.
	Returns:
		Map - Optimized map.
	"""
	# Creates PographObject
	pg = PoseGraph()

	# 'Translates' between the list-map to the object posegraph
	pg.importGraph(Map)

	# Optimizes the given map.
	pg.optimize(n_iter = 5)
	print('Graph optimized')

	# 'Translates' back the optimized graph to list-type.
	Map = pg.exportGraph(Map)

	return Map

def check_closure(Map, loop_step):

	"""
	Checks if loop closure has been achieved. If closure is detected adds the corresponding edges to
	the Map and the graph is optimized.
	Parameters:
		Map 	  - Map to check closure on.
		loop_step - Number of frames to skip before current frame.
	Returns:
		Map       - Map after loop closure check and optimization.
	"""
    
	# Map is unpacked.
	features, keyframes, nodes, edges = Map
	
	# Current frame features are obtained.
	_, des1 = features[-1]

	# Parameters are initialized.

	i = 0

	node_index = len(nodes)

	lc = False

	# Matcher is initialized.

	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks = 50)
	flann = cv2.FlannBasedMatcher(index_params, search_params)
		

	# All seen pictures except for the most recent 'loop_step' ones are checked.

	for target in features[:-loop_step]:

		# Compared features are unpacked and then matched.
		_, des2 = target

		matches = flann.knnMatch(des1,des2,k=2)

		good = []

		# Good features are selected as per Lowe's ratio.

		for m,n in matches:
			if m.distance < 0.75*n.distance:
				good.append([m])

		# If good features are more than 50 (needs to be changed to a variable threshold) then they are checked by a geometric check.

		if len(good) > 50:

			# Manhattan distance is computed.

			dist = np.abs(nodes[-1][0]-nodes[i][0])+np.abs(nodes[-1][1]-nodes[i][1])

			# If distance is less than 300 (another threshold must be introduced) then the loop is considered to be closed and the corresponding edge is added.

			if dist < 800:
				print('Loop closure detected')
				H     = homography(features[-1], features[i])
				edges.append([np.hstack((node_index,np.hstack((i+1,
										  np.array(H).flatten())))).T])
				lc    = True

		i+=1

	# Optimization is performed when closure is detected. 

	if lc:
		Map = optimize_map(Map)

	Map = [features, keyframes, nodes, edges]

	return Map

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

	path = np.asarray(path)
	path2 = np.roll(path, -1, axis=0)


	dx = path2[:,0]-path[:,0]
	dy = path2[:,1]-path[:,1]

	theta = np.arctan2(dy, dx)
	theta[-1] = theta[-2]
	path  = np.append(path, theta[:, np.newaxis], axis = 1)

	return path 

def move(map_name, path, window_size, plot = False, loop_step = 10, save_keyf = True):

	map_image = Image.open(map_name)

	if plot:
		map_image2 = cv2.cvtColor(np.array(map_image), cv2.COLOR_RGB2BGR)

	w = window_size[0]
	h = window_size[1]

	#Preprocess path in case it does not have theta information
	if np.shape(path)[1] != 3:
		path = pre_path(path)

	init = None

	steps_taken = 0


    #for pose in tqdm(path):
	for pose in path:
		y, x, alpha = pose

		if init == None:


			xc = x - w/2 * np.cos(alpha) - h/2 * np.sin(alpha)
			yc = y + w/2 * np.sin(alpha) - h/2 * np.cos(alpha)

			cropped_im = diagonal_crop.crop(map_image, (xc, yc), alpha, h, w)

			im1 = cv2.cvtColor(np.array(cropped_im), cv2.COLOR_RGB2BGR)

			#orig = [0,0]

			Map, init = add2map(im1, h, w, None, orig = [y, x], theta = alpha, save_keyf = save_keyf)

			im2 = im1

			if plot:
				plot_rectangles(map_image2, [x, y, alpha], [], h, w)
		else:

			steps_taken+=1

			if steps_taken % loop_step == 0:

				# print('Closure checking')
				Map = check_closure(Map, loop_step)


			xc = x - w/2 * np.cos(alpha) - h/2 * np.sin(alpha)
			yc = y + w/2 * np.sin(alpha) - h/2 * np.cos(alpha)

			cropped_im = diagonal_crop.crop(map_image, (xc, yc), alpha, h, w)

			im1 = cv2.cvtColor(np.array(cropped_im), cv2.COLOR_RGB2BGR)	

			Map, init = add2map(im1, im2, h, w, init = init, Map = Map, save_keyf = save_keyf)

			im2 = im1

			if plot:
				plot_rectangles(map_image2, [x,y,alpha], Map[2][-1], h, w)

	return Map

def orbgenerate(map_name, path, window_size, plot = False, loop_step = 10, save_keyf = True, folder = ''):

	map_image = Image.open(map_name)

	if plot:
		map_image2 = cv2.cvtColor(np.array(map_image), cv2.COLOR_RGB2BGR)

	w = window_size[0]
	h = window_size[1]

	#Preprocess path in case it does not have theta information
	if np.shape(path)[1] != 3:
		path = pre_path(path)

	init = None

	steps_taken = 0


	for pose in tqdm(path):
		#for pose in path:
		y, x, alpha = pose

		if init == None:


			xc = x - w/2 * np.cos(alpha) - h/2 * np.sin(alpha)
			yc = y + w/2 * np.sin(alpha) - h/2 * np.cos(alpha)

			cropped_im = diagonal_crop.crop(map_image, (xc, yc), alpha, h, w)

			im1 = cv2.cvtColor(np.array(cropped_im), cv2.COLOR_RGB2BGR)

			img_name = "{:06d}.png".format(steps_taken)

			img_name = os.path.join(folder , img_name)

			cv2.imwrite(img_name,im1)

			im2 = im1
			init = True

		else:

			steps_taken+=1

			xc = x - w/2 * np.cos(alpha) - h/2 * np.sin(alpha)
			yc = y + w/2 * np.sin(alpha) - h/2 * np.cos(alpha)

			cropped_im = diagonal_crop.crop(map_image, (xc, yc), alpha, h, w)

			im1 = cv2.cvtColor(np.array(cropped_im), cv2.COLOR_RGB2BGR)	

			img_name = "{:06d}.png".format(steps_taken)

			img_name = os.path.join(folder, img_name)

			cv2.imwrite(img_name,im1)

			im2 = im1

	z = np.arange(0,(steps_taken)/30,1/30)+1
	np.savetxt(os.path.join(folder[:-8] , 'times.txt'), z, fmt = '%10.6f' )
	return None


def capture(device, plot = False, loop_step = 10, save_keyf = True):

	video_capture = cv2.VideoCapture(device)

	init = None

	frames = 0

	while True:
		# Capture frame-by-frame
		ret, im1 = video_capture.read()

		frames += 1

		if not ret:
			break

		h, w = im1.shape[:2]

		if init == None:

			Map, init = add2map(im1, None, h, w, orig = [0,0], save_keyf = save_keyf)
			im2 = im1.copy()

		else:
			Map, init = add2map(im1, im2, h, w, init = init, Map = Map, save_keyf = save_keyf)
			im2 = im1.copy()

		if frames % loop_step == 0:

			print('Closure checking')
			Map = check_closure(Map, loop_step)

		if plot:
			cv2.imshow('frame', im1)

			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

	video_capture.release()
	cv2.destroyAllWindows()

	return Map

def locate2(Map, frame, h, w, curr_pos = None):

	surf = cv2.xfeatures2d.SURF_create()

	f1 = surf.detectAndCompute(frame,None)

	_, des1 = f1

	features, _, nodes, _ = Map

	i = 0

	pos_list = []

	if curr_pos is not None:

		posx0, posy0, theta0 = curr_pos
		nodes0 = np.asarray(nodes)
		mask = (np.abs(nodes0[:, 0]-posx0)+np.abs(nodes0[:, 1]-posy0))<500
		features = list(compress(features, mask.tolist()))
		nodes = list(compress(nodes, mask.tolist()))

	for target in features:

		_, des2 = target	

		matches = flann.knnMatch(des1,des2,k=2)

		good = []

		# Good features are selected as per Lowe's ratio.

		for m,n in matches:
			if m.distance < 0.75*n.distance:
				good.append([m])

		if len(good) > 40:


			H     = homography(f1, features[i])

			if H is not None:
			
				posx, posy, theta = nodes[i]

				alpha = H[0, 0]
				beta  = H[0, 1]


				H[0,2] = H[0,2]-((1-alpha)*w/2-beta*h/2)
				H[1,2] = H[1,2]-(beta*w/2+(1-alpha)*h/2)

				posx  += H[0,2]*np.cos(-theta)-H[1,2]*np.sin(-theta)
				posy  += H[0,2]*np.sin(-theta)+H[1,2]*np.cos(-theta)

				theta += np.arctan2(H[0, 1], H[0, 0])
				pos_list.append([posx, posy, theta])
		i += 1
	if len(pos_list)>1:
		meanx  = np.median(pos_list[:][0][0])
		meany  = np.median(pos_list[:][0][1])
		clean_list = []

		for pos in pos_list:
			x, y, theta = pos
			dist = np.abs(meanx-x)+np.abs(meany-y)

			if dist < 100:
				clean_list.append([x,y,theta])

		posx  = np.mean(clean_list[:][0][0])
		posy  = np.mean(clean_list[:][0][1])
		theta = np.mean(clean_list[:][0][2])
 

	else:
		return None, None

	return pos_list, [posx, posy, theta]

def localization_test2(Map, map_name, path, window_size, plot = 1):

	map_image = Image.open(map_name)

	map_image2 = cv2.cvtColor(np.array(map_image).copy(), cv2.COLOR_RGB2BGR)

	w = window_size[0]
	h = window_size[1]

	#Preprocess path in case it does not have theta information
	if np.shape(path)[1] != 3:
		path = pre_path(path)

	predicted_path = []
	position = None

    #for pose in tqdm(path):
	for pose in path:
		y, x, alpha = pose
		
		xc = x - w/2 * np.cos(alpha) - h/2 * np.sin(alpha)
		yc = y + w/2 * np.sin(alpha) - h/2 * np.cos(alpha)

		cropped_im = diagonal_crop.crop(map_image, (xc, yc), alpha, h, w)

		im1 = cv2.cvtColor(np.array(cropped_im), cv2.COLOR_RGB2BGR)

		pos_list, position = locate2(Map, im1, h, w, curr_pos = position)

		predicted_path.append(position)

		if plot:
			if position is not None:
				map_image2 = plot_rectangles(map_image2, [x, y, alpha], position, h, w, show = False)
			else:
				map_image2 = plot_rectangles(map_image2, [x, y, alpha], [], h, w, show = False)	
							
			cv2.imshow('window1', map_image2)
			cv2.waitKey(1)
		#print(np.asarray(pos_list), 'xpred')
		#print([x,y,alpha], 'xreal')



	return predicted_path

def locate(Map, frame, h, w):

	surf = cv2.xfeatures2d.SURF_create()

	f1 = surf.detectAndCompute(frame,None)

	_, des1 = f1

	features, _, nodes, _ = Map

	i = 0

	pos_list = []

	for target in features:

		_, des2 = target	

		matches = flann.knnMatch(des1,des2,k=2)

		good = []

		# Good features are selected as per Lowe's ratio.

		for m,n in matches:
			if m.distance < 0.75*n.distance:
				good.append([m])

		if len(good) > 40:

			H     = homography(f1, features[i])

			if H is not None:
			
				posx, posy, theta = nodes[i]

				alpha = H[0, 0]
				beta  = H[0, 1]


				H[0,2] = H[0,2]-((1-alpha)*w/2-beta*h/2)
				H[1,2] = H[1,2]-(beta*w/2+(1-alpha)*h/2)

				posx  += H[0,2]*np.cos(-theta)-H[1,2]*np.sin(-theta)
				posy  += H[0,2]*np.sin(-theta)+H[1,2]*np.cos(-theta)

				theta += np.arctan2(H[0, 1], H[0, 0])
				pos_list.append([posx, posy, theta])
		i += 1
	if len(pos_list)>1:

		meanx  = np.median(pos_list[:][0][0])
		meany  = np.median(pos_list[:][0][1])
		clean_list = []

		for pos in pos_list:
			x, y, theta = pos
			dist = np.abs(meanx-x)+np.abs(meany-y)

			if dist < 100:
				clean_list.append([x,y,theta])

		posx  = np.mean(clean_list[:][0][0])
		posy  = np.mean(clean_list[:][0][1])
		theta = np.mean(clean_list[:][0][2])
 

	else:
		return None, None

	return pos_list, [posx, posy, theta]

def localization_test(Map, map_name, path, window_size, plot = 1):

	map_image = Image.open(map_name)

	map_image2 = cv2.cvtColor(np.array(map_image).copy(), cv2.COLOR_RGB2BGR)

	w = window_size[0]
	h = window_size[1]

	#Preprocess path in case it does not have theta information
	if np.shape(path)[1] != 3:
		path = pre_path(path)

	predicted_path = []

    #for pose in tqdm(path):
	for pose in path:
		y, x, alpha = pose
		
		xc = x - w/2 * np.cos(alpha) - h/2 * np.sin(alpha)
		yc = y + w/2 * np.sin(alpha) - h/2 * np.cos(alpha)

		cropped_im = diagonal_crop.crop(map_image, (xc, yc), alpha, h, w)

		im1 = cv2.cvtColor(np.array(cropped_im), cv2.COLOR_RGB2BGR)

		pos_list, position = locate(Map, im1, h, w)

		predicted_path.append(position)

		if plot:
			if position is not None:
				map_image2 = plot_rectangles(map_image2, [x, y, alpha], position, h, w, show = False)
			else:
				map_image2 = plot_rectangles(map_image2, [x, y, alpha], [], h, w, show = False)	
							
			cv2.imshow('window1', map_image2)
			cv2.waitKey(1)
		#print(np.asarray(pos_list), 'xpred')
		#print([x,y,alpha], 'xreal')


	return predicted_path





