## overview
This model identifies snow depth from snow poles by identifying the top and bottom of the pole and converting the length in pixels to centimeters using a unique pixel to cm conversion for each camera. The pixel/cm ratio is derived using an image of the pole without snow, counting the pixels that represent the pole in the image, and dividing by the full length of the pole in centimeters (Breen et al. 2022). 

## data structure information
1) original images are saved in a nested subfolder from the root folder called "Data". Each camera folder has a unique folder ID that matches the camera ID. It is critical that the first part of the image name is the camera ID followed by an "_". For example, for camera E9E, an example image is E9E_0024.JPG, where E9E corresponds to the folder and the camera ID. 
2) annotations (x and y points) are saved as a .csv file from the root folder called "Data"

### Example images (image: left; mask: right)
![image](https://github.com/CV4EcologySchool/snow-Dayz/blob/main/snowpoles/example_imgs/E6A_WSCT0293.JPG)
![predict](https://github.com/CV4EcologySchool/snow-Dayz/blob/main/snowpoles/example_imgs/eval_E6A_WSCT0293.JPG.png)


## Training and evaluation
1) Before training on GPU, change the dataset root in the configuration files in `config` file. 

2) Train: 

on local or GPU machine: 
```
python train.py
```


3) Evaluate:
```
python evaluate.py 
```

## Basic packages:
- pytorch
- numpy


## other resources
Breen, C. M., C. Hiemstra, C. M. Vuyovich, and M. Mason. (2022). SnowEx20 Grand Mesa Snow Depth from Snow Pole Time-Lapse Imagery, Version 1 [Data Set]. Boulder, Colorado USA. NASA National Snow and Ice Data Center Distributed Active Archive Center. https://doi.org/10.5067/14EU7OLF051V. Date Accessed 04-13-2023.

