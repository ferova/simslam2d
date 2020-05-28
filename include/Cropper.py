import numpy as np
import cv2
from PIL import Image
from typing import Tuple, List
from include.Loader import Loader
from include.utils.map_utils import pre_path
from include import diagonal_crop
from include.utils.plot_utils import plot_rectangles
from include.utils.traj_utils import create_traj
import matplotlib.pyplot as plt

class Cropper:
	def __init__(
		self,
		image_path: str,
		trajectory_path: str = 'lisajous',
		crop_resolution: Tuple = (480,640),
		loader_resolution: Tuple = (5000, 5000),
		plot: bool = True,
		augmenter = None
		):

		self.image_path = image_path
		self.trajectory_path = trajectory_path
		self.crop_resolution = crop_resolution
		self.loader_resolution = loader_resolution



		self.count = 0
		self.loader = Loader(image_path = image_path,
							 resolution = crop_resolution,
							 area_to_load = loader_resolution)
		self.image = None


		self.trajectory = create_traj(trajectory_path, self.loader.ymax, self.loader.xmax)


		plt.plot(self.trajectory[:, 0],self.trajectory[:, 1])
		plt.axis('equal')
		plt.show()


		if self.trajectory.shape[1] < 3:
			self.trajectory = pre_path(self.trajectory)


		self.aug = augmenter
		self.plot = plot

	def __iter__(self):
			self.count = 0
			return self

	def __next__(self):

		if self.count+1 > len(self.trajectory):
			raise StopIteration

		aug = self.aug

		count  = self.count
		pose = self.trajectory[count]
		x, y, alpha = pose
		loader = self.loader
		h, w = self.crop_resolution


		image = loader.load(pose)

		xc = x - w/2 * np.cos(alpha) - h/2 * np.sin(alpha)
		yc = y + w/2 * np.sin(alpha) - h/2 * np.cos(alpha)

		# Diagonal crop the loaded image.

		if image is not None:
			crop = diagonal_crop.crop(Image.fromarray(image), (xc - loader.current_corner[0], yc - loader.current_corner[1]), alpha, h, w)
			self.image = image
		else:
			crop = diagonal_crop.crop(Image.fromarray(self.image), (xc - loader.current_corner[0], yc - loader.current_corner[1]), alpha, h, w)

		# Perform augmentation if any.

		if aug is not None:
			crop = aug(self, np.array(crop))

		# Plot the loaded area together with the part cropped.

		if self.plot:
			canvas = plot_rectangles(self.image, [x-loader.current_corner[0],y-loader.current_corner[1], alpha], [], h, w, show = False)
			cv2.namedWindow('canvas', cv2.WINDOW_NORMAL)
			cv2.resizeWindow('canvas', 600,600)
			cv2.imshow('canvas', canvas)
			cv2.waitKey(1)
		
		self.count += 1

		# Ensure the crop has the required resolution by cropping or padding, this is due to diagonal_crop rounding.

		crop = np.array(crop)

		ch, cw = crop.shape[0], crop.shape[1]

		c = np.zeros((h, w, crop.shape[2]), dtype=crop.dtype )

		if ch > h:
			crop = crop[:h, :, :]
			ch = h

		if cw > w:
			crop = crop[:, w, :]
			cw = w

		c[:ch,:cw,:] = crop

		return c

