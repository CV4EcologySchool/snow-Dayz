'''
###Generic function to load images

We will resize the image and set to float32. In the loop we will separate 
the binary testing data into two separate binary images, which are 
technically just inverted masks of each other. Then, we will put them 
into one tensor using stack so that the tensor is 128 x 128 x 2 channels

'''


import tensorflow as tf
import tensorflow_datasets as tfds

def load_image(image, mask):
  input_image = tf.image.resize(image, (128, 128))
  input_mask = tf.image.resize(mask, (128, 128))
  input_image = tf.cast(input_image, tf.float32) / 255.0
  return input_image, input_mask