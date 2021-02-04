import sys, getopt
import cv2
from tqdm import tqdm
from include.Cropper import Cropper
from imgaug import augmenters as iaa
import numpy as np


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

   def mb_augmenter(cropper, img):
      count  = cropper.count
      pose = cropper.trajectory[count]
      x, y, alpha = pose


      seq = iaa.Sequential([
               iaa.MotionBlur(k=[3,21], angle=np.rad2deg(alpha-np.pi/2))
               #iaa.GammaContrast(1.25)
               #iaa.GaussianBlur(sigma=(1, 3.0))
               #iaa.Cutout(fill_mode="constant", cval=0, size = 0.5)
               ])


      imglist = []

      imglist.append(img)

      imglist = seq.augment_images(imglist)

      return imglist[0]


   cropper = Cropper(inputfile, trajfile, (480, 720), (2000, 2000), augmenter = mb_augmenter)

   itercropper = iter(cropper)
   
   i = 0
   for img in tqdm(cropper):
      #cv2.imwrite(outputfile+str(i)+'.png',img)
      cv2.imshow('', img)
      cv2.waitKey(500)
      i+=1
if __name__ == "__main__":
   main(sys.argv[1:])
   # python simslamAugExample.py -i data/concrete_ours_picture.hdf5 -t test