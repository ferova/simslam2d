import numpy as np
import cv2
from typing import Tuple, List
from Loader import Loader
from utils.map_utils import pre_path
import diagonal_crop
from PIL import Image

class Cropper:
	def __init__(
		self,
		image_path: str,
		trajectory_path: str,
		crop_resolution: Tuple = (640,480),
		loader_resolution: Tuple = (5000, 5000)
		):

		self.image_path = image_path
		self.trajectory_path = trajectory_path
		self.crop_resolution = crop_resolution
		self.loader_resolution = loader_resolution

		self.trajectory = np.genfromtxt(trajectory_path, delimiter = ',')
		print(self.trajectory.shape)
		if self.trajectory.shape[1] < 3:
			self.trajectory = pre_path(self.trajectory)

		self.count = 0
		self.loader = Loader(image_path = image_path,
							 resolution = crop_resolution,
							 area_to_load = loader_resolution)
		self.image = None

	def __iter__(self):
			self.count = 0
			return self

	def __next__(self):

		if self.count+1 > len(self.trajectory):
			raise StopIteration


		count  = self.count
		pose = self.trajectory[count]
		x, y, alpha = pose
		loader = self.loader
		h, w = self.crop_resolution

		image = loader.load(pose)

		xc = x - w/2 * np.cos(alpha) - h/2 * np.sin(alpha)
		yc = y + w/2 * np.sin(alpha) - h/2 * np.cos(alpha)


		if image is not None:
			crop = diagonal_crop.crop(Image.fromarray(image), (xc - loader.current_corner[0], yc - loader.current_corner[1]), alpha, h, w)
			print('newly loaded')
			self.image = image
		else:

			crop = diagonal_crop.crop(Image.fromarray(self.image), (xc - loader.current_corner[0], yc - loader.current_corner[1]), alpha, h, w)

		cv2.imshow('canvas', self.image)
		cv2.waitKey(1)
		
		self.count += 1

		return np.array(crop)


#Open with Loader

cropper = Cropper('/home/laboratorio/simslam2d/stitched_tile.hdf5','/home/laboratorio/simslam2d/trajectory.csv', (500, 500), (5000, 5000))
itercropper = iter(cropper)

for img in cropper:
	cv2.imshow('', img)
	cv2.waitKey(1)

'''
#Open part of image
from PIL import Image
with h5py.File('/home/laboratorio/simslam2d/stitched_granite.hdf5', 'r') as f:
	xmax = f['stitched'].shape[0]
	ymax = f['stitched'].shape[1]
	print(f['stitched'].shape)
	img = Image.fromarray(f['stitched'][0:10000, 0:4731, :])
	img.show()
'''