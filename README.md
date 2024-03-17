# snow-Dayz

This github respository is to facilitate weather detection in camera trap images. 

### Examples of model predictions

<img src="https://github.com/CV4EcologySchool/snow-Dayz/blob/main/exp_imgs/Picture1.jpg">

## background

Changing winter weather will have an impact on snow conditions, but climate change research has focused far more on changing temperatures than precipitation. Specifically, mid-winter rain events are becoming more common in high latitude regions as a result of sporadic warm, humid days (Cohen et al., 2015). These mid-winter rain events can have drastic impacts in hydrology and ecology. In snow hydrology, precipitation is necessary to calibrate snow models for water availability estimates, and the wrong precipitation type in the early season can lower model accuracy for the rest of the winter season (Raleigh et al., 2016, 2015). In ecology, different winter precipitation types have different effects on animals. Deep snow following snowstorms may favor snow-adapted predators, increasing the predator’s likelihood for prey capture. Rain falling on snow (which is likely during mid-winter rain events) can create an icy layer, restricting vegetation access for grazing ungulates (Hansen et al., 2019). Alternatively, rain falling on snow can melt the snowpack, potentially reducing energetic costs for winter movement. Winter weather can also affect the detection probability estimate necessary for occupancy model development (Poley et al., 2018). Here, we present a model to detect weather conditions at camera traps. 

## model prediction
After downloading this repository onto your local machine. An example to predict weather in your images of interest is to use the command line is as follows: 

on local or GPU machine:

```
python classifier/predict.py --exp_name [model folder] --images_folder [image folder]
```

- 'exp_name' folder for CNN model 
- 'images_folder' folder for images that you would like to test the model on. 

Please reach out for any questions or comments or feedback. 