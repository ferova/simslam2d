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
      'inputfolder': 'image_path',
      'trajectory.name': 'trajectory_path',
      'trajectory.res': 'trajectory_res',
      'cropper.res': 'crop_resolution',
      'loader.res': 'loader_resolution',
      'plot.traj': 'plot_traj',
      'plot.crop': 'plot' 
}

def main(argv):
   inputfile = ''
   outputfile = ''
   try:
      opts, args = getopt.getopt(argv,"h:c:",["cfile="])
   except getopt.GetoptError:
      print('python simslam.py -c <conffile>')
      sys.exit(2)

   if not opts:
      print('python simslam.py -c <conffile>')
      sys.exit(2)

   for opt, arg in opts:
      if opt == '-h':
         print('python simslam.py -c <conffile>')
         sys.exit()
      elif opt in ("-c", "--cfile"):
         conffile = arg

   with open(os.path.abspath(os.path.join(os.getcwd(), conffile))) as file:
      config = yaml.full_load(file)
   
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
               cropperconf['crop_resolution'] = (v, config.get('crop_area.y')) 
            else:
               assert IOError('Failed to read crop_area.y')
         elif k == 'crop_area.y':
            if config.get('crop_area.x'):
               cropperconf['crop_resolution'] = (config.get('crop_area.x'), v) 
            else:
               assert IOError('Failed to read crop_area.y')

   cropper = Cropper(**cropperconf)

   #https://stackoverflow.com/questions/3061/calling-a-function-of-a-module-by-using-its-name-a-string

   for img in tqdm(cropper):
            #img_name = "{:06d}.jpg".format(i)

            #img_name = os.path.join(folder, img_name)
            #print(img_name)
            #cv2.imwrite(img_name, img)
            cv2.imshow('', img)
            cv2.waitKey(1)
            #i+=1


"""
def main(argv):
   inputfile = ''
   outputfile = ''
   try:
      opts, args = getopt.getopt(argv,"h:i:o:t:",["ifile=","ofile="])
   except getopt.GetoptError:
      print('python simslam.py -i <inputfile> -t <trajectoryfile> -o <outputfile>')
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print('python simslam.py -i <inputfile> -t <trajectoryfile> -o <outputfile>')
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputfile = arg
      elif opt in ("-o", "--ofile"):
         outputfile = arg
      elif opt in ("-t"):
         trajfile = arg

   #textures = ['asphalt_led', 'concrete_ours_picture', 'concrete_ours_vid','wood']
   textures = ['wood']
   trajectories = ['lisajous']
   #trajectories = ['squircle']
   #scales = [0.017, 0.05, 0.012, 0.017]
   for texture in textures:
      for trajectory in trajectories:

         print('Currently doing: ', texture, trajectory)


         cropper = Cropper(os.path.join('data/', texture+'.hdf5'), trajectory, (720, 1280), (4000, 4000))
         print(cropper.loader.xmax)
         print(cropper.loader.ymax)
         continue
         itercropper = iter(cropper)

         x, y, theta = cropper.trajectory[:, 0], cropper.trajectory[:, 1], cropper.trajectory[:, 2]

         plt.plot(x,y)
         plt.axis('equal')
         plt.show()
         ratios = []
         for i in range(len(x)-1):
            ratios.append(get_overlap_ratio([x[i],y[i],theta[i]], [x[i+1], y[i+1], theta[i+1]], 480, 720))

         traj_statistics = {}

         traj_statistics['overlap_ratio'] = {'mean':sum(ratios)/len(ratios), 'max':max(ratios), 'min':min(ratios)}

         print(traj_statistics)

         folder = 'C:\\Users\\jrodri56\\Documents\\GitHub\\simslam2d\\data\\test1\\'+texture+'\\'+trajectory+'\\'
         
         with open(os.path.join(folder, trajectory+'.txt'),'w+') as f:
            for xi, yi, ti in zip(x,y,theta):
               c, s = np.cos(ti), np.sin(ti)
               R = np.matrix('{} {}; {} {}'.format(c, -s, s, c))

               f.write('{:06f} {:06f} {:06f} {:06f} {:06f} {:06f} {:06f} {:06f} {:06f} {:06f} {:06f} {:06f}\n'.format(c, -s, 0, xi, s, c, 0, yi, 0, 0, 1, 0))

         with open('metadata.json', 'w+') as outfile:
            json.dump(os.path.join(folder, trajectory+'.txt'), outfile)



         i=0

         for img in tqdm(cropper):
            img_name = "{:06d}.jpg".format(i)

            img_name = os.path.join(folder, img_name)
           # print(img_name)
            cv2.imwrite(img_name, img)
            cv2.imshow('', img)
            cv2.waitKey(1)
            i+=1
"""
if __name__ == "__main__":
   main(sys.argv[1:])
   # python simslam.py -i data/concrete_ours_picture.hdf5 -t trajectory.csv