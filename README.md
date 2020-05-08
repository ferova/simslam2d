Simslam2D

Software to simulate what a robot sees when walking and looking down the floor on a given trajectory. The inputs are a trajectory of poses in csv format (can be only the xy poses) and an image stored in hdf5 format.

In order to use it you can run 

`python simslam.py -i <inputfile> -t <trajectoryfile> -o <outputfile>`

where inputfile is the input hdf5, trajectoryfile is the input csv and outputfile is the output directory to store the images and the poses to.

In order to easily convert the panorama from image file to hdf5 run 

`utils/im2hdf5.py -i <inputfile> -o <outputfile>`

In order to manipulate the image in some ways, please read how simslam.py and Cropper.py are implemented.

Documentation is pending.

This software uses the following libraries:

numpy
matplotlib
opencv-python
scipy
olefile
Pillow
h5py
tqdm
diagonal-crop (https://github.com/jobevers/diagonal-crop)

If this program has helped you in you research feel free to cite it:

```latex
@inproceedings{rodriguez2019simslam,
  title={SimSLAM 2D: A Simulation Framework for Testing and Benchmarking of two-dimensional Visual-SLAM Methods},
  author={Rodriguez, Juan and Castano-Cano, Davinson},
  booktitle={2019 19th International Conference on Advanced Robotics (ICAR)},
  pages={141--147},
  year={2019},
  organization={IEEE}
}
```