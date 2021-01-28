import sys, getopt
import cv2
from tqdm import tqdm
import numpy as np
import os
from simslam2d.Cropper import Cropper
from simslam2d.utils.traj_utils import get_overlap_ratio
import json 
import matplotlib.pyplot as plt
import yaml

conf2cropper = {
      'inputfile': 'image_path',
      'trajectory.name': 'trajectory_path',
      'trajectory.res': 'trajectory_res',
      'cropper.res': 'crop_resolution',
      'loader.res': 'loader_resolution',
      'plot.traj': 'plot_traj',
      'plot.croparea': 'plot' 
}

def main(argv):
   inputfile = ''
   outputfile = ''
   try:
      opts, args = getopt.getopt(argv,"h:c:",["cfile="])
   except getopt.GetoptError:
      print('python main.py -c <conffile>')
      sys.exit(2)

   if not opts:
      print('python main.py -c <conffile>')
      sys.exit(2)

   for opt, arg in opts:
      if opt == '-h':
         print('python main.py -c <conffile>')
         sys.exit()
      elif opt in ("-c", "--cfile"):
         conffile = arg

   with open(os.path.abspath(os.path.join(os.getcwd(), conffile))) as file:
      config = yaml.full_load(file)
   
   # Configuration parameters are parsed

   cropperconf = {}
   
   for k,v in config.items():
      if conf2cropper.get(k):
         cropperconf[conf2cropper[k]] = v
      else:
         if k == 'load_area.x':
            if config.get('load_area.y'):
               cropperconf['loader_resolution'] = (v, config.get('load_area.y')) 
            else:
               assert IOError('Failed to read load_area.y')
         elif k == 'load_area.y':
            if config.get('load_area.x'):
               cropperconf['loader_resolution'] = (config.get('load_area.x'), v) 
            else:
               assert IOError('Failed to read load_area.y')

         if k == 'crop_area.x':
            if config.get('crop_area.y'):
               cropperconf['crop_resolution'] = (config.get('crop_area.y'), v) 
            else:
               assert IOError('Failed to read crop_area.y')
         elif k == 'crop_area.y':
            if config.get('crop_area.x'):
               cropperconf['crop_resolution'] = (v, config.get('crop_area.x')) 
            else:
               assert IOError('Failed to read crop_area.y')

   # Augmentation parameters are parsed.

   if config['augmentation']:
      from imgaug import augmenters as iaa
      augmenter_list = []

      for aug in config['augmenters']:
         aug_placeholder = getattr(iaa, aug['augmenter'])
         del aug['augmenter']
         augmenter_list.append(aug_placeholder(**aug))

      seq = iaa.Sequential(augmenter_list)

   cropper = Cropper(**cropperconf)

   i = 0
   
   # Images are saved and displayed.

   if config['augmentation']:
      for img in tqdm(cropper):
         img = seq.augment_image(img)

         if config['saveimages']:
            img_name = "{:010d}.jpg".format(i)
            img_name = os.path.join(config.get('outputfolder', 'data/outputimages/'), img_name)
            cv2.imwrite(img_name, img)

         if config.get('plot.crop', False):
            cv2.imshow('Crop', img)
            cv2.waitKey(1)
         i+=1
   else:
      for img in tqdm(cropper):
         
         if config.get('saveimages', False):
            img_name = "{:010d}.jpg".format(i)
            img_name = os.path.join(config.get('outputfolder', 'data/outputimages/'), img_name)
            cv2.imwrite(img_name, img)

         if config.get('plot.crop', False):
            cv2.imshow('Crop', img)
            cv2.waitKey(1)

         i+=1


   x, y, theta = cropper.trajectory[:, 0], cropper.trajectory[:, 1], cropper.trajectory[:, 2]

   if config.get('trajname'):
      if config['trajname'][-4:] == '.txt':
         trajpath = config['trajname']
      else:
         trajpath = config['trajname']+'.txt'
   else:
      trajpath = 'estimated_trajectory.txt'

   with open(os.path.join(trajpath),'w+') as f:
      for xi, yi, ti in zip(x,y,theta):
         c, s = np.cos(ti), np.sin(ti)
         R = np.matrix('{} {}; {} {}'.format(c, -s, s, c))

         f.write('{:06f} {:06f} {:06f} {:06f} {:06f} {:06f} {:06f} {:06f} {:06f} {:06f} {:06f} {:06f}\n'.format(c, -s, 0, xi, s, c, 0, yi, 0, 0, 1, 0))

if __name__ == "__main__":
   main(sys.argv[1:])