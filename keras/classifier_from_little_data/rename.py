#!/usr/bin/env python

'''This script goes along the blog post
"Building powerful image classification models using very little data"
from blog.keras.io.
It uses data that can be downloaded at:
https://www.kaggle.com/c/dogs-vs-cats/data
In our setup, we:
- created a data/ folder
- created train/ and validation/ subfolders inside data/
- created cats/ and dogs/ subfolders inside train/ and validation/
- put the cat pictures index 0-999 in data/train/cats
- put the cat pictures index 1000-1400 in data/validation/cats
- put the dogs pictures index 12500-13499 in data/train/dogs
- put the dog pictures index 13500-13900 in data/validation/dogs
So that we have 1000 training examples for each class, and 400 validation examples for each class.
In summary, this is our directory structure:
```
data/
    train/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
    validation/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
```
'''
import glob
import os
import shutil

def key_func(x):
    return int(os.path.basename(x).split('.')[1])

'''
for i, f in zip(range(100000), sorted(glob.glob('./origin/cat*.jpg'), key=key_func)):
    if i < 1000: 
        new_name = "./train/cats/cat%03d.%s" % (i, os.path.basename(f).split('.')[2]) 
    elif i >= 1000 and i < 1400:
        new_name = "./validation/cats/cat%03d.%s" % (i, os.path.basename(f).split('.')[2]) 
    elif i >= 1400:
        break

    print ("[%s => [%s]" % (f, new_name))
    shutil.copy(f, new_name)
'''

for i, f in zip(range(100000), sorted(glob.glob('./origin/dog*.jpg'), key=key_func)):
    if i < 1000: 
        new_name = "./train/dogs/dog%03d.%s" % (i, os.path.basename(f).split('.')[2]) 
    elif i >= 1000 and i < 1400:
        new_name = "./validation/dogs/dog%03d.%s" % (i, os.path.basename(f).split('.')[2]) 
    elif i >= 1400:
        break

    print ("[%s => [%s]" % (f, new_name))
    shutil.copy(f, new_name)

