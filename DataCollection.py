import cv2
from Hand import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)

offset = 20
imgSize = 300

folder = r"D:\Pycharm\ASL2H\Dataset\A"
counter = 0

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        x_min = min(hand['bbox'][0] for hand in hands)
        y_min = min(hand['bbox'][1] for hand in hands)
        x_max = max(hand['bbox'][0] + hand['bbox'][2] for hand in hands)
        y_max = max(hand['bbox'][1] + hand['bbox'][3] for hand in hands)

        x, y, w, h = x_min - offset, y_min - offset, x_max - x_min + 2 * offset, y_max - y_min + 2 * offset
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        imgCrop = img[max(0, y):min(img.shape[0], y + h), max(0, x):min(img.shape[1], x + w)]

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    # if key == ord("s"):
    while counter<10:
        counter += 1
        cv2.waitKey(1)
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)
    counter=0
