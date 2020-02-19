import cv2

def draw(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def draw_bounds(img, mask):
    pass