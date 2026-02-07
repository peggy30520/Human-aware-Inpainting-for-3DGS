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
save_mask_dir = "../gaussian-splatting/demo/mask_train"
save_masked_dir = "../gaussian-splatting/demo/masked_image_train"

if not os.path.exists(save_mask_dir):
    os.makedirs(save_mask_dir)

if not os.path.exists(save_masked_dir):
    os.makedirs(save_masked_dir)

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
results = model(images_list[:index], save = True, classes = [0, 24], show_boxes = False, show_conf = False, show_labels = False, device ='0', imgsz = (h*0.8, w*0.8), retina_masks = True) # predict on an image

# Access the results
for i, result in enumerate(results):
    if result.masks == None:
        total_mask = np.zeros((h, w, 3))
        cv2.imwrite(os.path.join(save_mask_dir, image_list[i].split('.')[0]+'.png'), total_mask)
    else:
        masks = result.masks.data.cpu().numpy()
        total_mask = masks[0]

        for j, mask in enumerate(masks):
            if j == 0:
                continue
            else:
                total_mask += mask
        
        total_mask = np.clip(total_mask, 0, 1)
        total_mask = np.uint8(255 * total_mask)
        total_mask = cv2.dilate(total_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pad,pad)))
        cv2.imwrite(os.path.join(save_mask_dir, image_list[i].split('.')[0]+'.png'), total_mask)
    
        # Combine the orginal image and mask to check the quality of detection
        img = cv2.imread(images_list[i])
        white = np.full((h, w, 3), 255)
        white = cv2.bitwise_and(white, white, mask=total_mask)
        total_mask = cv2.bitwise_not(total_mask)
        masked = cv2.bitwise_and(img, img, mask=total_mask)
        masked = masked + white
        cv2.imwrite(os.path.join(save_masked_dir, image_list[i]), masked)

results.clear()
torch.cuda.empty_cache()

# Predict with the model (second batch)
results = model(images_list[index:], save = True, classes = [0, 24], show_boxes = False, show_conf = False, show_labels = False, device ='0', imgsz = (h*0.8, w*0.8), retina_masks = True) # predict on an image

# Access the results
for i, result in enumerate(results):
    if result.masks == None:
        total_mask = np.zeros((h, w, 3))
        cv2.imwrite(os.path.join(save_mask_dir, image_list[i+index].split('.')[0]+'.png'), total_mask)

    else:
        masks = result.masks.data.cpu().numpy()
        total_mask = masks[0]

        for j, mask in enumerate(masks):
            if j == 0:
                continue
            else:
                total_mask+=mask
        
        total_mask = np.clip(total_mask, 0, 1)
        total_mask = np.uint8(255 * total_mask)
        total_mask = cv2.dilate(total_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pad,pad)))
        cv2.imwrite(os.path.join(save_mask_dir, image_list[i+index].split('.')[0]+'.png'), total_mask)
    
        # Combine the orginal image and mask to check the quality of detection
        img = cv2.imread(images_list[i+index])
        white = np.full((h, w, 3), 255)
        white = cv2.bitwise_and(white, white, mask=total_mask)
        total_mask = cv2.bitwise_not(total_mask)
        masked = cv2.bitwise_and(img, img, mask=total_mask)
        masked = masked + white
        cv2.imwrite(os.path.join(save_masked_dir, image_list[i+index]), masked)
    
