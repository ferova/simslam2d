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
		self.image_transformations = [np.array(t.split(), np.float32()) for t in database[1::2]]
		self.count = 0

		self.x_centers = [a[2] for a in self.image_transformations]
		self.y_centers = [a[5] for a in self.image_transformations]


		self.max_x = max(self.x_centers)
		self.max_y = max(self.y_centers)

		self.min_x = min(self.x_centers)
		self.min_y = min(self.y_centers)

		print(self.min_x)

	def __iter__(self):
		return self

	def __next__(self):

		cv2.namedWindow('window1' , cv2.WINDOW_NORMAL)

		h = self.max_y - self.min_y
		w = self.max_x - self.min_x

		print(h,w)
		for i in range(len(self.image_paths)//2): 

			im_src = cv2.imread(self.image_paths[i])
			transform = self.image_transformations[i]
			transform = np.reshape(transform, (-1,3))

			transform[0,2]+=abs(self.min_x)
			#transform[1,2]+=self.min_y
			print(transform[0,2])

			if i == 0:
				im_dst = cv2.warpPerspective(im_src, np.float32(transform), (int(w//2), int(h)), borderMode=cv2.BORDER_TRANSPARENT)
			else:

				im_dst = cv2.warpPerspective(im_src, np.float32(transform), (im_dst.shape[1], im_dst.shape[0]), im_dst, borderMode=cv2.BORDER_TRANSPARENT)


			# Display images
			cv2.imshow("window1", im_dst)
			cv2.resizeWindow('window1', 1500,1500)
			cv2.imshow("window1", im_dst)

			cv2.waitKey(1)
		cv2.waitKey(0)				 


loader = Loader('/home/jfrvk/Documents/Simslam2D/databases/granite/', '')
myiter = iter(loader)

next(myiter)