import numpy as np
import cv2
from PIL import Image
from typing import Tuple, List
from include.Loader import Loader
from include.utils.map_utils import pre_path
from include import diagonal_crop
from include.utils.plot_utils import plot_rectangles
import matplotlib.pyplot as plt

class Cropper:
	def __init__(
		self,
		image_path: str,
		trajectory_path: str = 'test',
		crop_resolution: Tuple = (640,480),
		loader_resolution: Tuple = (5000, 5000),
		plot: bool = True,
		augmenter = None
		):

		self.image_path = image_path
		self.trajectory_path = trajectory_path
		self.crop_resolution = crop_resolution
		self.loader_resolution = loader_resolution

		if self.trajectory_path != 'test':
			self.trajectory = np.genfromtxt(trajectory_path, delimiter = ',')
		else:
			t = np.linspace(0, 2*np.pi+0.01, 100)

			x = 2500*np.cos(3*t)+3000
			y = 1000*np.sin(t)  +2000

			self.trajectory = np.append(y[:, np.newaxis], x[:, np.newaxis], axis=1)*0.5


		if self.trajectory.shape[1] < 3:
			self.trajectory = pre_path(self.trajectory)

		self.count = 0
		self.loader = Loader(image_path = image_path,
							 resolution = crop_resolution,
							 area_to_load = loader_resolution)
		self.image = None

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


		if image is not None:
			crop = diagonal_crop.crop(Image.fromarray(image), (xc - loader.current_corner[0], yc - loader.current_corner[1]), alpha, h, w)
			self.image = image
		else:
			crop = diagonal_crop.crop(Image.fromarray(self.image), (xc - loader.current_corner[0], yc - loader.current_corner[1]), alpha, h, w)

		if aug is not None:
			crop = aug(crop)

		if self.plot:
			canvas = plot_rectangles(self.image, [x-loader.current_corner[0],y-loader.current_corner[1], alpha], [], h, w, show = False)
			cv2.namedWindow('canvas', cv2.WINDOW_NORMAL)
			cv2.resizeWindow('canvas', 600,600)
			cv2.imshow('canvas', canvas)
			cv2.waitKey(1)
		
		self.count += 1

		return np.array(crop)

