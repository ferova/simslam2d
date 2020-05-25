import sys, getopt
import cv2
from tqdm import tqdm
from include.Cropper import Cropper
import numpy as np
import os

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

   cropper = Cropper(inputfile, trajfile, (480, 720), (4000, 4000))

   itercropper = iter(cropper)

   x, y, theta = cropper.trajectory[:, 0], cropper.trajectory[:, 1], cropper.trajectory[:, 2]

   #trajectory = cropper.trajectory

#   trajectory = np.roll(trajectory, -1, axis=0) - trajectory
   folder = 'C:\\Users\\jrodri56\\Documents\\GitHub\\simslam2d\\data\\test1\\asphalt_led\\'+trajfile+'\\'
   
   with open(os.path.join(folder, trajfile+'.txt'),'w+') as f:
      for xi, yi, ti in zip(x,y,theta):
         c, s = np.cos(ti), np.sin(ti)
         R = np.matrix('{} {}; {} {}'.format(c, -s, s, c))

         f.write('{:06f} {:06f} {:06f} {:06f} {:06f} {:06f} {:06f} {:06f} {:06f} {:06f} {:06f} {:06f}\n'.format(c, -s, 0, xi, s, c, 0, yi, 0, 0, 1, 0))

   i=0

   for img in tqdm(cropper):
      break
      img_name = "{:06d}.png".format(i)

      img_name = os.path.join(folder, img_name)
     # print(img_name)
      cv2.imwrite(img_name, img)
      cv2.imshow('', img)
      cv2.waitKey(1)
      i+=1

if __name__ == "__main__":
   main(sys.argv[1:])
   # python simslam.py -i data/concrete_ours_picture.hdf5 -t trajectory.csv