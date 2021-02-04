import numpy as np
import cv2
import h5py
from typing import Tuple, List
import logging


class Loader:
	def __init__(
		self,
		image_path: str,
		resolution: Tuple,
		area_to_load: Tuple
		):
		self.path = image_path
		self.res  = resolution
		self.area = area_to_load
		self.current_corner = [0, 0]
		self.initial_load = False
		try:
			with h5py.File(self.path, 'r') as f:
				self.xmax = f['stitched'].shape[1]
				self.ymax = f['stitched'].shape[0]
		except Exception as e:
			raise IOError from e
	def load(self, pose):

		x, y, _ = pose
		ah, aw = self.area

		xmax = self.xmax
		ymax = self.ymax

		if xmax < aw or ymax < ah:
			raise ValueError('One of the dimensions in the loading area is larger than those of the terrain.')
		
		
		if self.isInArea(pose, self.res) and self.initial_load:
			return None
		else:
			with h5py.File(self.path, 'r') as f:

				if int(x+aw//2) < xmax and int(y+ah//2) < ymax:

					if int(x-aw//2) < 0:
						if int(y - ah//2) < 0:
							logging.debug('case 1')
							self.current_corner = [0, 0]
							image = f['stitched'][0:ah, 0:aw]
						else:
							logging.debug('case 2')
							self.current_corner = [0, int(y-ah//2)]
							image = f['stitched'][int(y-ah//2):int(y+ah//2), 0:aw]
					else:
						if int(y - ah//2) < 0:
							logging.debug('case 3')
							self.current_corner = [int(x-aw//2), 0]
							image = f['stitched'][0:ah,int(x-aw//2):int(x+aw//2)]
						else:
							logging.debug('case 4')
							self.current_corner = [x-aw//2, y-ah//2]
							image = f['stitched'][int(y-ah//2):int(y+ah//2), int(x-aw//2):int(x+aw//2)]

				elif int(x+aw//2) > xmax and int(y+ah//2) > ymax:
					logging.debug('case 5')
					self.current_corner = [xmax-aw, ymax-ah]
					image = f['stitched'][ymax-ah:ymax, xmax-aw:xmax]

				elif int(x+aw//2) > xmax:
					if int(y - ah//2) < 0:
						logging.debug('case 6')
						self.current_corner = [xmax-aw, 0]
						image = f['stitched'][0:ah, xmax-aw:xmax]
					else:
						logging.debug('case 7')
						self.current_corner = [xmax-aw, y-ah//2]
						image = f['stitched'][int(y-ah//2):int(y+ah//2), xmax-aw:xmax]

				else:
					if int(x - aw//2) < 0:
						logging.debug('case 8')
						self.current_corner = [0, ymax-ah]
						image = f['stitched'][ymax-ah:ymax, 0:aw]
					else:
						logging.debug('case 9')
						self.current_corner = [x-aw//2, ymax-ah]
						image = f['stitched'][ ymax-ah:ymax, int(x-aw//2):int(x+aw//2)]
		self.initial_load = True

		return image

	def isInArea(self, pose, resolution):

		x, y, alpha = pose
		h, w    = resolution

		xmin = self.current_corner[0]
		ymin = self.current_corner[1]

		xmax = self.current_corner[0]+self.area[0]
		ymax = self.current_corner[1]+self.area[1]

		points = []

		points.append([x - w/2 * np.cos(alpha) - h/2 * np.sin(alpha), y + w/2 * np.sin(alpha) - h/2 * np.cos(alpha)])
		points.append([x + w/2 * np.cos(alpha) - h/2 * np.sin(alpha), y - w/2 * np.sin(alpha) - h/2 * np.cos(alpha)])
		points.append([x + w/2 * np.cos(alpha) + h/2 * np.sin(alpha), y - w/2 * np.sin(alpha) + h/2 * np.cos(alpha)])
		points.append([x - w/2 * np.cos(alpha) + h/2 * np.sin(alpha), y + w/2 * np.sin(alpha) + h/2 * np.cos(alpha)])

		for pt in points:
			if xmin < pt[0] < xmax:
				if ymin < pt[1] < ymax:
					pass
				else:
					return False
			else:
				return False
		return True

	
