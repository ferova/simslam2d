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

   cropper = Cropper(inputfile, trajfile, (480, 720), (5000, 5000))

   itercropper = iter(cropper)
   
   i = 0

   folder = data/test1

   for img in tqdm(cropper):

      img_name = "{:06d}.png".format(i)

      img_name = os.path.join(folder , img_name)

      cv2.imwrite(img_name,im1)
      #cv2.imwrite(outputfile+str(i)+'.png',img)
      cv2.imshow('', img)
      cv2.waitKey(1)
      i+=1

if __name__ == "__main__":
   main(sys.argv[1:])
   # python simslam.py -i data/concrete_ours_picture.hdf5 -t trajectory.csv