import cv2
import h5py
import sys, getopt

def main(argv):
   inputfile = ''
   outputfile = ''
   try:
      opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
   except getopt.GetoptError:
      print('im2hdf5.py -i <inputfile> -o <outputfile>')
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print('test.py -i <inputfile> -o <outputfile>')
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputfile = arg
      elif opt in ("-o", "--ofile"):
         outputfile = arg

   im = cv2.imread(inputfile)

   with h5py.File(outputfile, "w") as f:
      f.create_dataset("stitched", dtype = np.uint8, data = im.view(np.uint8), chunks = True)

if __name__ == "__main__":
   main(sys.argv[1:])