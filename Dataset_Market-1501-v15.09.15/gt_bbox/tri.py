import json
import cv2
import numpy as np
import os

from pprint import pprint
from os import listdir
from os.path import isfile, join
from shutil import copyfile

mypath = '.'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

#for file_i in onlyfiles:
#    if (file_i != 'tri.py') and (file_i != 'Thumbs.db'):
#        cam_and_obj = file_i.split('_')
#        cam_i = cam_and_obj[1].split('c')[1].split('s')[0]
#        obj_i = cam_and_obj[0]
#
#        print(cam_i, obj_i)
#
#        os.makedirs('./C'+cam_i+'/O'+obj_i, exist_ok=True)
#        copyfile(file_i, './C'+cam_i+'/O'+obj_i+'/'+file_i)




#for file_i in onlyfiles:
#    if (file_i != 'tri.py') and (file_i != 'Thumbs.db'):
#        cam_and_obj = file_i.split('_')
#        cam_i = cam_and_obj[1].split('c')[1].split('s')[0]
#        obj_i = cam_and_obj[0]
#
#        os.makedirs('./market1501/'+obj_i, exist_ok=True)
#        copyfile(file_i, './market1501/'+obj_i+'/'+file_i)

nums = []
for i in range(1502):
    nums.append(0)

for file_i in onlyfiles:
    if (file_i != 'tri.py') and (file_i != 'Thumbs.db'):
        cam_and_obj = file_i.split('_')
        cam_i = cam_and_obj[1].split('c')[1].split('s')[0]
        obj_i = cam_and_obj[0]

        os.makedirs('./market/val/'+obj_i, exist_ok=True)
        os.makedirs('./market/train/'+obj_i, exist_ok=True)

        if nums[int(obj_i)]%5 == 0:
            copyfile(file_i, './market/train/'+obj_i+'/'+file_i)
        else:
            copyfile(file_i, './market/val/'+obj_i+'/'+file_i)
            nums.append(obj_i)

        nums[int(obj_i)] = nums[int(obj_i)] + 1
