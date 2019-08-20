import cv2
import numpy as np
from scipy import ndimage
from keras.models import load_model


class IdCardReader:
    def __init__(self):
        self.id_number_length = 12
        self.card_model = load_model('models/card_segmentation_119epochs.h5', compile=False)
        self.digit_model = load_model('models/digit_100epochs.h5', compile=False)

    # Main flow
    def extract_id_number(self, file_path):
        try:
            # Read image from file path
            img_org = cv2.imread(file_path)

            # Predict mask
            mask = self.predict(img_org)

            # Detect id card area
            img = self.crop_id_card(img_org, mask)

            # Crop id number from card
            img_id_number = self.crop_id_number(img)

            id_number = self.read_id_number(img_id_number)
            return id_number
        except Exception as e:
            print(str(e))
            return ''

    def predict(self, img_org):
        # Resize to 512x512
        img = cv2.resize(img_org, (512, 512))
        img = img / 255.0

        # Predict mask using unet model
        predict = self.card_model.predict(img.reshape(1, 512, 512, 3))

        # Get mask
        mask = predict[0]
        mask = mask * 255.0
        mask = mask.astype(np.uint8)
        return mask

    def crop_id_card(self, img_org, mask):
        # Open
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (70, 70))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Close
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Thresh with 100
        mask = cv2.threshold(mask, 150, 255, cv2.THRESH_BINARY)[1]

        # Find contours and get contour max
        contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
        contour_max = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(contour_max)
        box = cv2.boxPoints(rect).astype(np.int32)

        # Fill color for mask with coordinates
        mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [box], color=(255, 255, 255))
        mask = cv2.resize(mask, (img_org.shape[1], img_org.shape[0]))

        # Bitwise original image and mask
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
        return img

    def crop_id_number(self, img):
        w, h = img.shape[1], img.shape[0]
        tl_x, tl_y = int(w * 2 / 6), int(h / 4)
        br_x, br_y = int(w * 7 / 8 + w / 14), int(h / 3 + h / 14)
        img_id_number = img[tl_y:br_y, tl_x:br_x]
        return img_id_number

    def read_id_number(self, img_id_number):
        # Convert to gray
        img_id_number_gray = cv2.cvtColor(img_id_number, cv2.COLOR_BGR2GRAY)
        img_id_number_gray = cv2.adaptiveThreshold(img_id_number_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                   cv2.THRESH_BINARY, 41, 7)

        # Close
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        img_id_number_gray = cv2.morphologyEx(img_id_number_gray, cv2.MORPH_CLOSE, kernel)

        # Processing and bitwise
        img_id_number_gray = cv2.GaussianBlur(img_id_number_gray, (5, 5), 0)
        img_id_number_gray = cv2.threshold(img_id_number_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        img_id_number_gray = cv2.bitwise_not(img_id_number_gray)

        # Find contours
        contours_hs = []
        contours = cv2.findContours(img_id_number_gray, cv2.RETR_LIST, cv2.RETR_CCOMP)[1]
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

        # Sorting contours by x position
        id_number = []
        contours = sorted(contours, key=lambda x: self.get_x(x))
        for contour in contours:
            rect = cv2.boundingRect(contour)
            x, y, w, h = rect
            img_number = img_id_number_gray[y:y + h, x:x + w]
            img_number = cv2.resize(img_number, (29, 44))
            digit_pred = self.digit_model.predict(img_number.reshape(1, 44, 29, 1))
            id_number.append(str(np.argmax(digit_pred)))
            # cv2.rectangle(img_id_number, (x, y), (x + w, y + h), (0, 0, 255), 1)

        return ''.join(id_number)

    def get_x(self, x):
        rect = cv2.boundingRect(x)
        x, y, w, h = rect
        return x
