from ultralytics import YOLO
import numpy as np
import os
import cv2
import torch
from PIL import Image

# Load a model
model = YOLO("yolo11x-seg.pt")

# Set the data path 
img_dir = "../gaussian-splatting/demo/images"
save_masked_dir = "../gaussian-splatting/demo/bounding_box_images"
save_mask_txt_dir = "../gaussian-splatting/demo/bounding_box_txt"

if not os.path.exists(save_masked_dir):
    os.makedirs(save_masked_dir)

if not os.path.exists(save_mask_txt_dir):
    os.makedirs(save_mask_txt_dir)


image_list = os.listdir(img_dir)
images_list = []
for image in image_list:
    images_list.append(os.path.join(img_dir, image))

img = cv2.imread(images_list[0])
h, w, _ = img.shape

# Sometimes the images are too big, split to two batches to deal with detection
index = int(len(image_list)/2)

# To ensure the mask will contain the edge of human 
pad = 15

# Predict with the model (first batch)
results = model(images_list[:index], save = True, classes = [0, 24], show_boxes = True, show_conf = False, show_labels = False, device ='0', imgsz = (h*0.8, w*0.8), retina_masks = True) # predict on an image

# Access the results
for i, result in enumerate(results):
    f  = open(os.path.join(save_mask_txt_dir, image_list[i].split('.')[0]+'.txt'), "w")
    if result.boxes == None:
        f.close()
    else:
        boxes = result.boxes.xyxy.cpu().numpy() 
        img = cv2.imread(images_list[i])
        #print(boxes)
        for box in boxes:
            s_p = (int(box[0]) - pad, int(box[1]) - pad)
            e_p = (int(box[2]) + pad, int(box[3]) + pad)
            #print(s_p, e_p)
            img = cv2.rectangle(img, s_p, e_p, (0, 255, 0), 5)
            L = str(max(0, int(box[0]) - pad))
            R = str(min(w, int(box[2]) + pad))
            T = str(max(0, int(box[1]) - pad))
            B = str(min(h, int(box[3]) + pad))
            f.write(L + "," + R + "," + T + "," + B + "\n")
        
        # Combine the orginal image and bounding box to check the quality of detection
        cv2.imwrite(os.path.join(save_masked_dir, image_list[i]), img)
        f.close()

results.clear()
torch.cuda.empty_cache()

# Predict with the model (second batch)
results = model(images_list[index:], save = True, classes = [0, 24], show_boxes = False, show_conf = False, show_labels = False, device ='0', imgsz = (h*0.8, w*0.8), retina_masks = True) # predict on an image

# Access the results
for i, result in enumerate(results):
    f  = open(os.path.join(save_mask_txt_dir, image_list[i+index].split('.')[0]+'.txt'), "w")
    if result.boxes == None:
        f.close()
    else:
        boxes = result.boxes.xyxy.cpu().numpy() 
        img = cv2.imread(images_list[i+index])
        #print(boxes)
        for box in boxes:
            s_p = (int(box[0]) - pad, int(box[1]) - pad)
            e_p = (int(box[2]) + pad, int(box[3]) + pad)
            #print(s_p, e_p)
            img = cv2.rectangle(img, s_p, e_p, (0, 255, 0), 5)
            L = str(max(0, int(box[0]) - pad))
            R = str(min(w, int(box[2]) + pad))
            T = str(max(0, int(box[1]) - pad))
            B = str(min(h, int(box[3]) + pad))
            f.write(L + "," + R + "," + T + "," + B + "\n")
        
        # Combine the orginal image and bounding box to check the quality of detection
        cv2.imwrite(os.path.join(save_masked_dir, image_list[i+index]), img)
        f.close()
    
