import cv2
import numpy as np
class Loader:
	def __init__(
		self,
		database_file: str,
		trajectory_file: str,
	):
		self.database_file = database_file + 'database.txt'
		self.trajectory_file = trajectory_file

		with open(self.database_file) as f:
			database = f.read().splitlines()
		self.image_paths = [database_file + s for s in database[::2]]
		self.image_transformations = database[1::2]

		self.count = 0

	def __iter__(self):
		return self

	def __next__(self):

		im_dst = np.zeros((10000,10000,3), np.uint8)

		for i in range(1): 

			cv2.namedWindow('window1' , cv2.WINDOW_NORMAL)
			im_src = cv2.imread(self.image_paths[i])
			transform = np.array(self.image_transformations[i].split())
			transform = np.reshape(transform, (-1,3))

			im_dst = cv2.warpPerspective(im_src, np.float32(transform), (im_dst.shape[1], im_dst.shape[0]))

			# Display images
			#cv2.imshow("window1", im_dst)
			cv2.imshow("Destination Image", im_dst)
			cv2.resizeWindow('window1', 1500,1500)

			cv2.waitKey(0)
			 


loader = Loader('/home/jfrvk/Documents/Simslam2D/databases/granite/', '')
myiter = iter(loader)

next(myiter)