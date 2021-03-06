# Simslam 2D

![Simslam2D Demp](demo/demo.gif)

Simslam2D is a research-oriented, open-source software to simulate what a robot sees when traversing a planar environment while looking down the floor on a given trajectory. It is completely written in Python with the use of OpenCV, Numpy and HDF5.

This opens up the possibility of rapidly prototyping new planar visual-SLAM algorithms via simulation.

## Features

- Can output images in real time.

- Multiple default trajectories as well as custom ones.

- Can work on any panoramic picture of the ground.

- Provides trajectory ground-truth in KITTI format.

- Data augmentation can be applied to images using [imgaug](https://github.com/aleju/imgaug).

## Dependencies

Please install dependencies before running with `pip install -r requirements.txt`

- numpy
- matplotlib
- opencv-python
- h5py
- tqdm
- pyyaml

# Tutorials

## Preprocessing

### Micro GPS

1. Install all the necessary depdencies with `pip`:

    `pip install -r requirements.txt`

2. Download and extract one of the databases of Micro GPS at https://microgps.cs.princeton.edu/ .

3. Run the preprocessor script by calling:


    `python simslam2d/Preprocessor.py -i <inputfolder> -o <outputfile>`

Here, the inputfolder has to be the extracted database folder, the outputfile has to be the path to the desired panorama, it must have the .hdf5 extension. For example, if one downloads the granite database and extracts it to `data/granite` calling the following:
`python simslam2d/Preprocessor.py -i data/granite/ -o data/granite.hdf5` would produce a panorama called granite.hdf5 in the data folder.

### Panoramic images

1. Install all the necessary depdencies with `pip`:

    `pip install -r requirements.txt`

2. Have your panoramic picture as ONE image file readable by OpenCV.

3. Run the preprocessor script by calling:

    `python simslam2d/im2hdf5.py -i <inputfile> -o <outputfile>`

Here, the inputfile is a panoramic image of the terrain. For example, calling

`python simslam2d/im2hdf5.py -i panorama.png -o data/outputfile.hdf5` would produce a panorama called outputfile.hdf5 in the data folder.

#### Example

1. Download the example panorama [here](https://drive.google.com/uc?export=download&id=1VGKm3iKqvPX6j75ADiIrq6D-YO4aWV_T).

2. Run the preprocesing script:

    `python simslam2d/im2hdf5.py -i 'data/MVI_7756_stitch.png' -o data/concrete.hdf5`

This will create a terrain file suitable for use with Simslam 2D.

## Running a simulation

Once you have an input terrain in hdf5 format, you will also need a configuration file and, optionally, a trajectory file. We provide examples of configuration files in the conf folder.

In order to run the simulation with the previously generated images:

1. Update the configuration file in the inputfolder field to point to the path of the hdf5 panorama.
2. Run the following command:

    `python main.py -c conf/example1.yaml `


 __Note:__ In case you want to save the images you should also set the saveimages flag to *True* and the outputfolder to the folder you want to save the images to.

 For a description of all the parameters please refer to the table below.


| Parameter       | Type       | Description                                                  |
|-----------------|------------|--------------------------------------------------------------|
| inputfolder     | string     | The path to the terrain in hdf5 format.                      |
| saveimages      | bool       | Whether to save the resulting crops or not.                  |
| outputfolder    | string     | Folder where the resulting images are saved to.              |
| trajname        | string     | Output path to save the groundtruth to.                      |
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



\* The trajectory name can either be one of ['lisajous', 'squircle', 'sin2', 'layp'] or the path to a csv file with containing the poses in *x, y* pairs or *x,y,theta* triplets. Examples of these files can be found on the data folder. The poses in these files need to be given in pixel coordinates relative to the upper-left corner of the panorama.

** The list of augmenters consists of a list of dictionaries, in which each dictionary corresponds to an augmenter in the [imgaug library](https://imgaug.readthedocs.io/en/latest/source/api.html). Each dictionary must have the name of the augmenter in the _augmenter_ key and the parameters as keyword arguments. For example
a gaussian blur that has random sigma values from the set {1, 3, 9} is specified as {augmenter: GaussianBlur, sigma: [1, 3, 9]} in the augmenter list.

## Running a simulation with augmentation


In order to run the simulation with augmentation:

1. Set the augmentation field in the example configuration file to *True*.
2. Run the following command:

    `python main.py -c conf/example1.yaml `

In this case, we apply two augmenters. The first one is a Gamma contrast augmenter that chooses the gamma value randomly between 0.7, 0.8 and 0.9. The second augmenter is a Gaussian blur one that chooses a sigma value randomly between 1, 3, and 9. 

## Post-processing

For post-processing, we recommend using the [evo](https://github.com/MichaelGrupp/evo) post-processor for comparing the groundtruth provided by Simslam 2D and the trajectory estimated by the SLAM algorithm. Note our trajectory is in kitti format.

# Citation

If this program has helped you please consider citing it:

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