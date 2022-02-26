# Crowd Counting with Deep Learning (Mini Project)

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
