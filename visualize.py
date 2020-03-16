import cv2
import numpy as np


def create_mask():
    img = cv2.imread('ting.jfif')
    cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), (0, 0, 0), -1)
    cv2.rectangle(img, (0, 0), (140, 170), (255, 255, 255), -1)
    maskedimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return maskedimg


def draw_bounds(img, mask):
    processed = cv2.imread(img)
    fg = cv2.bitwise_and(processed, processed, mask=mask)
    cv2.imshow('image', fg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def iterate():
    a = [[1, 2, 3], [4, 5, 6]]
    b = np.array(a)
    for x in xrange(b.shape[0]):
        for y in yrange(b.shape[1]):
            if b[x][y] == 1: img[x][y][1] += 30
            if b[x][y] == 2: img[x][y][0] += 30
            if b[x][y] == 3: img[x][y][2] += 30
            if b[x][y] == 4:
                img[x][y][0] += 30
                img[x][y][1] += 30
            if b[x][y] == 5:
                img[x][y][0] += 30
                img[x][y][2] += 30
            if b[x][y] == 6:
                img[x][y][1] += 30
                img[x][y][2] += 30
            #if b[x][y] == 7:


def test():
    img = cv2.imread('test\\ting.jfif')
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_img[100:300,000:600,0] += 75
    tint_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    cv2.imshow('image', tint_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

test()
# 'car': 1,
# 'person': 2,
# 'rider': 3,
# 'bus': 4,
# 'bike': 5,
# 'motor': 6,
# 'truck': 7,
# 'traffic light': 8,
# 'train': 9,
# 'traffic sign': 10,
# https://www.youtube.com/watch?v=CCOXg75HkvM