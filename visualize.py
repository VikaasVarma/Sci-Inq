from PIL import Image
from transforms import RGBTransform


def iterate(d, pic):
    """Iterate through a provided image and find regions that should be tinted.
    Input should be a dictionary where the key is whatever object is bound by
    the region, and the value is a tuple containing the two corners of the
    bounding region.
    """
    for key in d:
        x = d[key[0]], y = d[key[1]], x1 = d[key[2]], y1 = d[key[3]]
        if key == "car":
            tint(x, y, x1, y1, (255, 0, 0), pic)
        elif key == "person":
            tint(x, y, x1, y1, (0, 255, 0), pic)
        elif key == "rider":
            tint(x, y, x1, y1, (0, 0, 255), pic)
        elif key == "bus":
            tint(x, y, x1, y1, (255, 255, 77), pic)
        elif key == "bike":
            tint(x, y, x1, y1, (149, 0, 179), pic)
        elif key == "motor":
            tint(x, y, x1, y1, (102, 255, 255), pic)
        elif key == "truck":
            tint(x, y, x1, y1, (255, 102, 255), pic)
        elif key == "traffic light":
            tint(x, y, x1, y1, (102, 51, 0), pic)
        elif key == "train":
            tint(x, y, x1, y1, (0, 128, 0), pic)
        elif key == "traffic sign":
            tint(x, y, x1, y1, (255, 0, 0), pic)
        else:
            # some other color for generic objects
            tint(x, y, x1, y1, (0, 0, 0), pic)


def tint(x, y, x1, y1, color, img):
    """This method will tint an image a provided color in the region given.
    X and Y should be the coordinates for the top left corner of the region,
    and x1 and y1 are coordinates for the bottom right corner. The color should
    be given in the form of a tuple of length 3 with RGB values.
    """
    # pic = cv2.imread('test\\ting.jfif')
    pic = Image.open(img)
    tinted = RGBTransform().mix_with(
        color, factor=.30).applied_to(pic.crop((x, y, x1, y1)))
    pic.paste(tinted, (x, y))
    #pic.show()
    return pic

#tint(0, 200, 300, 500, (255, 0, 0), 'test\\ting.jfif')
# 'car': 1,
# 'person': 2,
# 'rider': 3,
# 'bus': 4,
# 'bike': 5,
# 'motor': 6,
# 'truck': 7,
# 'traffic light': 8
# 'train': 9,
# 'traffic sign': 10,
# https://www.youtube.com/watch?v=CCOXg75HkvM
