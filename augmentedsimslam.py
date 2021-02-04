import sys, getopt
import cv2
from tqdm import tqdm
from include.Cropper import Cropper
import numpy as np
import os
from include.utils.traj_utils import get_overlap_ratio
import json 
from imgaug import augmenters as iaa

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
   textures = [ 'concrete_ours_picture','wood']
   trajectories = ['squircle', 'layp']
   #trajectories = ['squircle']

   for texture in textures:
      for trajectory in trajectories:

         print('Currently doing: ', texture, trajectory)


         cropper = Cropper(os.path.join('data/', texture+'.hdf5'), trajectory, (720, 1280), (4000, 4000))

         itercropper = iter(cropper)

         x, y, theta = cropper.trajectory[:, 0], cropper.trajectory[:, 1], cropper.trajectory[:, 2]



         ratios = []

         for i in range(len(x)-1):
            ratios.append(get_overlap_ratio([x[i],y[i],theta[i]], [x[i+1], y[i+1], theta[i+1]], 480, 720))

         traj_statistics = {}

         traj_statistics['overlap_ratio'] = {'mean':sum(ratios)/len(ratios), 'max':max(ratios), 'min':min(ratios)}

         print(traj_statistics)

         folder = 'C:\\Users\\jrodri56\\Documents\\GitHub\\simslam2d\\data\\test2\\'
         with open(os.path.join(folder, trajectory+'.txt'),'w+') as f:
            for xi, yi, ti in zip(x,y,theta):
               c, s = np.cos(ti), np.sin(ti)
               R = np.matrix('{} {}; {} {}'.format(c, -s, s, c))

               f.write('{:06f} {:06f} {:06f} {:06f} {:06f} {:06f} {:06f} {:06f} {:06f} {:06f} {:06f} {:06f}\n'.format(c, -s, 0, xi, s, c, 0, yi, 0, 0, 1, 0))

         with open('metadata.json', 'w+') as outfile:
            json.dump(os.path.join(folder, trajectory+'.txt'), outfile)



         i=0
         mb_params = [19, 23, 25]#[3, 7, 11, 15, 19]
         gb_params = [2, 3, 5]#[0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
         for img in tqdm(cropper):

            img_name = "{:06d}.jpg".format(i)
            for k, sigma  in zip(mb_params, gb_params):

               alpha = theta[i]

               #mb_aug = iaa.MotionBlur(k=k, angle=0)#-np.pi/2))
               gb_aug = iaa.GaussianBlur(sigma=sigma)

               foldermb = folder+'\\'+'motion_blur'+'\\'+str(k)+'\\'+texture+'\\'+trajectory
               foldergb = folder+'\\'+'gaussian_blur'+'\\'+str(sigma)+'\\'+texture+'\\'+trajectory

               finalmb_name = os.path.join(foldermb, img_name)
               finalgb_name = os.path.join(foldergb, img_name)

               #print(finalmb_name)
               #print(finalgb_name)

               #mb_img = mb_aug.augment_image(img)
               gb_img = gb_aug.augment_image(img)
               
               cv2.imwrite(finalgb_name, gb_img)
               #cv2.imwrite(finalmb_name, mb_img)

               #cv2.imshow('mb', mb_img)
               cv2.imshow('gb', gb_img)
               cv2.waitKey(1)
            
            i+=1

if __name__ == "__main__":
   main(sys.argv[1:])
   # python simslam.py -i data/concrete_ours_picture.hdf5 -t trajectory.csv