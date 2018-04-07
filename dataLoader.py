import os
import cv2
import random


def loadData(args, img_size):

    obj_cats = os.path.join(os.getcwd(),'101_ObjectCategories')

    categories = args
    if args[0] == 'ALL':
        categories = os.listdir(obj_cats)
    labeled_data = []
    cat_num = 0
    num_cats = len(categories)
    for cat in categories:
        path_head = os.path.join(obj_cats, cat) 
        for file_name in os.listdir(path_head):
            path = os.path.join(path_head, file_name)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (img_size, img_size))
            output = [0]*num_cats
            output[cat_num] = 1
            labeled_data.append((img, output))
        cat_num = cat_num + 1

    random.shuffle(labeled_data)
    return labeled_data