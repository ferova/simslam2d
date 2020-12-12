### Simslam2D

Software to simulate what a robot sees when walking and looking down the floor on a given trajectory. The inputs are a trajectory of poses in csv format (can be only the xy poses) and an image stored in hdf5 format.

In order to use it you can run 

`python simslam.py -i <inputfile> -t <trajectoryfile> -o <outputfile>`

where inputfile is the input hdf5, trajectoryfile is the input csv and outputfile is the output directory to store the images and the poses to. For a test trajectory type 'test' in the trajectoryfile field.

In order to easily convert the panorama from image file to hdf5 run 

`utils/im2hdf5.py -i <inputfile> -o <outputfile>`

In order to manipulate the image in some ways, please read how simslam.py and Cropper.py are implemented.

### MicroGPS Preprocessing Tutorial

1. Install all the necessary depdencies with `pip`:
`pip install -r requirements.txt`
2. Download and extract one of the databases of Micro GPS at https://microgps.cs.princeton.edu/ .
3. Run the preprocessor script by calling:
`python simslam2d/Preprocessor.py -i <inputfolder> -o <outputfile>`

Here, the inputfolder has to be the extracted database folder, the outputfile has to be the path to the desired panorama, it must have the .hdf5 extension. For example, if one downloads the granite database and extracts it to `data/granite` calling the following:
`python simslam2d/Preprocessor.py -i data/granite/ -o data/granite.hdf5` would produce a panorama called granite.hdf5 in the data folder.

### Running Simslam2d 


| Parameter       | Type       | Description                                                  |
|-----------------|------------|--------------------------------------------------------------|
| inputfolder     | string     | The path to the terrain in hdf5 format.                      |
| saveimages      | bool       | Whether to save the resulting crops or not.                  |
| outputfolder    | string     | Folder where the resulting images are saved to.              |
| load_area.x     | int        | Width of the loaded area.                                    |
| load_area.y     | int        | Height of the loaded area.                                   |
| crop_area.x     | int        | Width of the resulting cropped images.                       |
| crop_area.y     | int        | Height of the resulting cropped images.                      |
| trajectory.name | string     | Name of the trajectory or path to csv. *                     |
| trajectory.res  | int        | Number of poses in the trajectory.                           |
| plot.traj       | bool       | Plot a preview of the trajectory.                            |
| plot.croparea   | bool       | Plot the loaded area with the relative position of the crop. |
| plot.crop       | bool       | Plot the resulting crop.                                     |
| augmentation    | bool       | Whether or not to apply                                      |
| augmenters      | list(dict) | List of augmenter definitions. **                            |

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