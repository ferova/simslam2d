import sys, getopt
import cv2
from tqdm import tqdm
from include.Cropper import Cropper

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

   cropper = Cropper(inputfile,trajfile, (500, 500), (2000, 2000))
   itercropper = iter(cropper)
   
   for img in tqdm(cropper):
      cv2.imwrite(outputfile+str())
      cv2.imshow('', img)
      cv2.waitKey(1)

if __name__ == "__main__":
   main(sys.argv[1:])
   # python simslam.py -i concrete_ours.hdf5 -t trajectory.csv