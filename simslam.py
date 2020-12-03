import sys, getopt
import cv2
from tqdm import tqdm
from include.Cropper import Cropper
import numpy as np
import os
from include.utils.traj_utils import get_overlap_ratio
import json 
import matplotlib.pyplot as plt

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

if __name__ == "__main__":
   main(sys.argv[1:])
   # python simslam.py -i data/concrete_ours_picture.hdf5 -t trajectory.csv