import numpy as np
import cv2
import h5py
from typing import Tuple, List
import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)

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

	def load(self, pose):

		x, y, _ = pose
		aw, ah = self.area

		with h5py.File(self.path, 'r') as f:

			#logging.debug(f['stitched'].shape)

			xmax = f['stitched'].shape[0]
			ymax = f['stitched'].shape[1]

		#logging.debug('maxx: {}, maxy: {}'.format(xmax, ymax))
		
		if self.isInArea(pose, self.res) and self.initial_load:
			logging.debug(self.isInArea(pose, self.res))
			self.isInArea(pose, self.res)
			logging.debug('isInArea')
			return None
		else:

			with h5py.File(self.path, 'r') as f:

				if int(x+aw//2) < xmax and int(y+ah//2) < ymax:

					if int(x-aw//2) < 0:
						if int(y - ah//2) < 0:
							logging.debug('case 1')
							self.current_corner = [0, 0]
							image = f['stitched'][0:aw, 0:ah]
						else:
							logging.debug('case 2')
							self.current_corner = [0, int(y-ah//2)]
							image = f['stitched'][0:aw, int(y-ah//2):int(y+ah//2)]
					else:
						if int(y - ah//2) < 0:
							logging.debug('case 3')
							self.current_corner = [int(x-aw//2), 0]
							image = f['stitched'][int(x-aw//2):int(x+aw//2), 0:ah]
						else:
							logging.debug('case 4')
							self.current_corner = [x-aw//2, y-ah//2]
							image = f['stitched'][int(x-aw//2):int(x+aw//2), int(y-ah//2):int(y+ah//2)]

				elif int(x+aw//2) > xmax and int(y+ah//2) > ymax:
					logging.debug('case 5')
					self.current_corner = [0, 0]
					image = f['stitched'][xmax-aw:xmax, ymax-ah:ymax]

				elif int(x+aw//2) > xmax:
					if int(y - ah//2) < 0:
						logging.debug('case 6')
						self.current_corner = [0, 0]
						image = f['stitched'][xmax-aw:xmax, 0:ah]
					else:
						logging.debug('case 7')
						self.current_corner = [0, 0]
						image = f['stitched'][xmax-aw:xmax, int(y-ah//2):int(y+ah//2)]

				else:
					if int(x - aw//2) < 0:
						logging.debug('case 8')
						self.current_corner = [0, 0]
						image = f['stitched'][0:aw, ymax-ah:ymax]
					else:
						logging.debug('case 9')
						self.current_corner = [0, 0]
						image = f['stitched'][int(x-aw//2):int(x+aw//2), ymax-ah:ymax]
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
			logging.debug(pt)
			logging.debug(xmin < pt[0] < xmax)
			logging.debug(ymin < pt[1] < ymax)
			if xmin < pt[0] < xmax:
				if ymin < pt[1] < ymax:
					pass
				else:
					return False
			else:
				return False

		return True

if __name__ == '__main__':
	#Open with Loader
	
	loader = Loader('/home/laboratorio/simslam2d/stitched_tile.hdf5', (1000, 500), (1500, 1500))
	for i in range(0, 100000, 1000):
		print(i, i)

		img = loader.load([i,i,0])
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