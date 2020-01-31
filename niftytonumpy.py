import nibabel as nib
import numpy as np
import os
import torch
import os
from os import listdir
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.spatial import distance
import numpy

import re
def sorted_aphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

x = []
y = []
for i in sorted_aphanumeric(os.listdir("/content/atlas304")): 
  if(i.find('.a') == 18):
    x.append(os.path.join('/content/atlas304',i))
  else:
    y.append(os.path.join('/content/atlas304',i))

train  = []
 for i in x:
   print(i)
   image = nib.load(f'{i}')
   image = image.get_fdata()
   image = np.array(image, dtype='uint8')
   train.append(image)

label = []
for i in y:
  print(i)
  image = nib.load(f'{i}')
  image = image.get_fdata()
  image = np.array(image, dtype='uint8')
  label.append(image)

train_file = train[:20]
test_file1 = train[20:40]
dataset = train[40:300]
train_label = label[:20]
test_label = label[20:40]
dataset_label = label[40:300]

train_files= 'train_images'
np.save(train_files+'.npy',train_file)

label_files = 'labels'
np.save(label_files+'.npy',train_label)

def flatten(t):
  t = torch.flatten(t)
  return t

comparing_dataset = {}
count = -1
for i in dataset:
  count+=1
  t = x[count]
  comparing_dataset[f'{t}'] = i.flatten()

similar_files = []
count = -1
for j in train_file:
  count+=1
  t = x[count]
  flatten_image = j.flatten()
  eucli = {}
  for i in comparing_dataset.keys():
    eucli[i]= distance.euclidean(flatten_image,comparing_dataset[i])
  similar_image = min(eucli, key=eucli.get)
  image = nib.load(f'{similar_image}')
  image = image.get_fdata()
  image = np.array(image, dtype='uint8')
  similar_files.append(image)


  print("Original Image",t+''+'    '+"Simmilar Image is:",min(eucli, key=eucli.get))

similar_file ="Similar_images"
np.save(similar_file+'.npy',similar_files)
