import cv2
import glob
import re

def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def fread(file_name):
    img_names = glob.glob(file_name)
    img_names = sorted(img_names, key=numericalSort)
    return img_names

def imread(file_name, flag=-1, scale=1.0):
    img_names = glob.glob(file_name)
    if len(img_names)==0:
        print("[No {} images]".format(file_name))
        exit()
    img_names = sorted(img_names, key=numericalSort)
    img_list = [cv2.resize(cv2.imread(path, flag), None, fx=scale, fy=scale) for path in img_names]
    return img_list
