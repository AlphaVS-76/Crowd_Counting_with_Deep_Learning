# Crowd Counting with Deep Learning (Mini Project)

## Pretrained CNN model

- [Model](https://github.com/AlphaVS-76/Mob_Counter/blob/main/pretrainedmodel.py)

  ### Methods followed
  
  - Non Maximum Suppression : is a key step in many computer vision applications. It is a class of algorithms to select one entity (e.g., bounding boxes) out of many overlapping entities. The most common approach for NMS for object detection is a greedy, locally optimal strategy with several hand-designed components (e.g., thresholds).

  - Confidence/Consistency Map : It basically is a probability density method to know how much an image is similar to the image prior to it. It assigns each pixel of the new image a probability, which is the probability of the pixel color occurring in the object in the previous image. More algorithms like this are ensemble tracking, CAMshifts, Kalman filter, mean-shift.
  - Gamma Correction : This controls the overall brightness of an image. Gamma values less than 1 will shift the image towards the darker end of the spectrum while gamma values greater than 1 will make the image appear lighter. 
  - Downscaling : Downscaling is any procedure to deduce high-resolution information from low-resolution images. This is used in case of images with low resolution.
  - CNN (Convolutional Neural Network) : This is a type of Neural network whose use is generally to analyze imagery. It uses a special technique called Convolution meaning it transforms an image using every pixel and their local neighbours. 
  - ReLU (Rectified Linear Unit) : It is an activation function which will output the input directly if it is positive, otherwise, it will output zero (Absolute classification). It is more effecient and time saving than Sigmoid function.


## Mounting
- Mount GDrive account to colab

## Imports

```
import numpy as np
import matplotlib.pyplot as plt
import pretrainedmodel as cnn
import cv2
import torch
```
## Images

- [Source](https://www.google.com/search?q=crowd&rlz=1C1VDKB_enIN964IN964&sxsrf=APq-WBsbQBLRPtfqFJS-3ABfPvtp2dIfcQ:1645932637709&source=lnms&tbm=isch&sa=X&ved=2ahUKEwjQhMfx-J72AhXzlFYBHSmrCf4Q_AUoAXoECAIQAw&biw=1536&bih=754&dpr=1.25)

## Importing the Data and Image

```
data = '/content/drive/MyDrive/weights/weights.pth'
```

```
image = cv2.imread('/content/crowd2.jpg')
```

## Detecting Heads

- To detect the heads and count them, use `.head_detection()` function.

## Counting

- Use `.sum()` function to count the heads in the image.
