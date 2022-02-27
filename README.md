# Crowd Counting with Deep Learning (Mini Project)

## Pretrained CNN model

- [Model](https://github.com/AlphaVS-76/Mob_Counter/blob/main/pretrainedmodel.py)

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
