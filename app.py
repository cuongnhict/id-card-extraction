import warnings
import cv2
import os
import numpy as np
from scipy import ndimage
from keras.models import load_model

warnings.filterwarnings('ignore')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def extract_id_number(file_path, model):
    img_org = cv2.imread(file_path)
    img = cv2.resize(img_org, (512, 512))
    img = img / 255.0

    predict = model.predict(img.reshape(1, 512, 512, 3))

    # Prediction mask
    mask = predict[0]
    mask = mask * 255.0
    mask = mask.astype(np.uint8)

    # Opening
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Thresh with 100
    mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)[1]

    # Find contours and get contour max
    contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
    contour_max = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(contour_max)
    box = cv2.boxPoints(rect).astype(np.int32)

    # Fill color for mask with coordinates
    mask = np.zeros(mask.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [box], color=(255, 255, 255))
    mask = cv2.resize(mask, (img_org.shape[1], img_org.shape[0]))

    # Bitwise
    img = cv2.bitwise_and(img_org, img_org, mask=mask)

    # Rotate image with angle
    img = ndimage.rotate(img, rect[2])

    # Crop image
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.threshold(img_gray, 10, 255, cv2.THRESH_BINARY)[1]
    contours = cv2.findContours(img_gray, cv2.RETR_LIST, cv2.RETR_CCOMP)[1]
    contour_max = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour_max)
    img = img[y:y + h, x:x + w]
    if w < h:
        img = ndimage.rotate(img, 90.0)
        w, h = h, w

    # Crop id
    top, left = int(w * 2 / 6), int(h / 4)
    bottom, right = int(w * 7 / 8 + w / 11), int(h / 3 + h / 11)
    img_id = img[left:right, top:bottom]

    # Convert to gray
    img_id_gray = cv2.cvtColor(img_id, cv2.COLOR_BGR2GRAY)
    img_id_gray = cv2.adaptiveThreshold(img_id_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 7)

    # Close
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img_id_gray = cv2.morphologyEx(img_id_gray, cv2.MORPH_CLOSE, kernel)

    # Processing and bitwise
    img_id_gray = cv2.GaussianBlur(img_id_gray, (5, 5), 0)
    img_id_gray = cv2.threshold(img_id_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    img_id_gray = cv2.bitwise_not(img_id_gray)

    # Find contours
    contours_hs = []
    contours = cv2.findContours(img_id_gray, cv2.RETR_LIST, cv2.RETR_CCOMP)[1]
    for contour in contours:
        rect = cv2.boundingRect(contour)
        x, y, w, h = rect
        if w < h and (w * h) > 300:
            contours_hs.append([contour, h])

    # Sort by h
    contours_hs = sorted(contours_hs, key=lambda x: x[1])
    std_contours_hs = []
    # Get std from coontours
    for idx in range(len(contours_hs)):
        if len(contours_hs) - idx >= 12:
            sub_contours_hs = contours_hs[idx: idx + 12]
            sub_hs = [i[1] for i in sub_contours_hs]
            std_contours_hs.append([idx, np.std(sub_hs)])

    # Get min std
    std_min = min(std_contours_hs, key=lambda x: x[1])
    # Get contours from std min index with 12 steps
    contours_hs = contours_hs[std_min[0]:std_min[0] + 12]
    contours = [i[0] for i in contours_hs]

    def get_x(x):
        rect = cv2.boundingRect(x)
        x, y, w, h = rect
        return x

    # Sorting contours by x position
    id_number = []
    contours = sorted(contours, key=lambda x: get_x(x))
    for contour in contours:
        rect = cv2.boundingRect(contour)
        x, y, w, h = rect
        img_number = img_id_gray[y:y + h, x:x + w]
        img_number = cv2.resize(img_number, (29, 44))
        digit_pred = digit_model.predict(img_number.reshape(1, 44, 29, 1))
        id_number.append(str(np.argmax(digit_pred)))
        # cv2.rectangle(img_id, (x, y), (x + w, y + h), (0, 0, 255), 1)

    return ''.join(id_number)


ID_NUMBER_LENGTH = 12
image_paths = [
    'images/mama_fullhd_00.jpg',
    'images/mama_fullhd_01.jpg',
    'images/mama_fullhd_02.jpg',
    'images/anh_fullhd_00.jpg',
    'images/cuong_fullhd_00.jpg',
    'images/cuong_fullhd_01.jpg',
    'images/cuong_fullhd_02.jpg',
    'images/hieu_fullhd_00.jpg',
    'images/hieu_fullhd_01.jpg',
    'images/hieu_fullhd_02.jpg',
    'images/huyen_fullhd_00.jpg',
    'images/huyen_fullhd_01.jpg',
    'images/huyen_fullhd_02.jpg',
    'images/luc_fullhd_00.jpg',
    'images/luc_fullhd_01.jpg'
]
model = load_model('model/card_segmentation_60epochs.h5', compile=False)
digit_model = load_model('model/digit_100epochs.h5', compile=False)

for file_path in image_paths:
    id_number = extract_id_number(file_path, model)

    image = cv2.imread(file_path)
    image = cv2.resize(image, (400, 400))
    cv2.putText(image, 'ID: ' + id_number, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imshow(file_path, image)
cv2.waitKey()
