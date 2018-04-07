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
        for i in range(1, len(os.listdir(path_head))+1):
            pad_dig = 4-len(str(i))
            pad = '0'*pad_dig
            file_name = 'image_' +pad+str(i)+'.jpg'
            path = os.path.join(path_head, file_name)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            print(img)
            img = cv2.resize(img, (img_size, img_size))
            output = [0]*num_cats
            output[cat_num] = 1
            labeled_data.append((img, cat_num))
        cat_num = cat_num + 1

    random.shuffle(labeled_data)
    return labeled_data