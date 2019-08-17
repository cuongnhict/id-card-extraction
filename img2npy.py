# Convert sample images to .npy file

import warnings
import cv2
import json
from glob import glob
import numpy as np

warnings.filterwarnings('ignore')


def get_image_and_mask(image_file_name, label_file_name):
    # Get image from image file name
    image = cv2.imread(image_file_name)
    # Get quad and coordinates from label_file_name
    quad = json.load(open(label_file_name, 'r'))
    coords = np.array(quad['quad'], dtype=np.int32)

    # Create mask same szie image
    mask = np.zeros(image.shape, dtype=np.uint8)
    # Fill color for mask with coordinates
    cv2.fillPoly(mask, [coords], color=(255, 255, 255))
    # Transform from bgr to gray
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    # Threshold mask
    _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)

    return image, mask


# Saved image and mask to .npy file
# Make X, y matrics to train
X = []
y = []
images_file_name = sorted(glob('dataset/transform/*.jpg'))
labels_file_name = sorted(glob('dataset/masks/*.json'))

i = 0

for image_file_name, label_file_name in zip(images_file_name, labels_file_name):
    image, mask = get_image_and_mask(image_file_name, label_file_name)
    X.append(image)
    y.append(mask)
    if i <= 10:
        cv2.imshow('image', image)
        cv2.imshow('mask', mask)
        cv2.waitKey()
    i += 1

X = np.array(X)
y = np.array(y)
y = np.expand_dims(y, axis=3)

# Save image and mask into .npy file
np.save('dataset/final_image.npy', X)
np.save('dataset/final_mask.npy', y)
print('Saved!')
