import numpy as np
import cv2
import h5py
from tqdm import tqdm

class Preprocessor:
	def __init__(
		self,
		database_file: str,
		trajectory_file: str,
		name: str = 'stitched_panorama'
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

		self.name = name

	def preprocess(self):


		h = self.max_y - self.min_y
		w = self.max_x - self.min_x


		for i in tqdm(range(len(self.image_transformations))): 
			
			im_src = cv2.imread(self.image_paths[i])
			transform = self.image_transformations[i]
			transform = np.reshape(transform, (-1,3))

			transform[0,2]+=abs(self.min_x)
			#transform[1,2]+=self.min_y
			

			if i == 0:
				im_dst = cv2.warpPerspective(im_src, np.float32(transform), (int(w//2)+1000, int(h)+1000), borderMode=cv2.BORDER_TRANSPARENT)
			else:

				im_dst = cv2.warpPerspective(im_src, np.float32(transform), (im_dst.shape[1], im_dst.shape[0]), im_dst, borderMode=cv2.BORDER_TRANSPARENT)
		
		with h5py.File(self.name+".hdf5", "w") as f:
			f.create_dataset("stitched", dtype = np.uint8, data = im_dst.view(np.uint8), chunks = True)




if __name__ == '__main__':
	preprocessor = Preprocessor('/home/laboratorio/simslam2d/databases/asphalt_led/', '', 'stitched_asphalt_led')
	preprocessor.preprocess()
