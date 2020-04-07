from algos import *
import pandas as pd
import numpy as np
import os
from data import Data
from algos import *



def ensemble(start, end):
    d = Data()
    path = "D\\se2\\Sci-Inq"
    lsst = []
    class1 = pd.DataFrame([], columns = list('01')) 
    class2 = pd.DataFrame([], columns = list('01')) 
    class3 = pd.DataFrame([], columns = list('01')) 
    class4 = pd.DataFrame([], columns = list('01')) 
    class5 = pd.DataFrame([], columns = list('01')) 
    class6 = pd.DataFrame([], columns = list('01')) 
    class7 = pd.DataFrame([], columns = list('01')) 
    class8 = pd.DataFrame([], columns = list('01')) 
    class9 = pd.DataFrame([], columns = list('01'))
    class10 = pd.DataFrame([], columns = list('01'))
    class11 = pd.DataFrame([], columns = list('01'))
    class12 = pd.DataFrame([], columns = list('01'))
    d.get_data('train', start, end)

    for i in range(start, end):
        img = d.x[i]
        imga = img.astype(np.uint8)
        imgb = img.astype(np.unit16)
        stats = [brightness(imgb), fourier_sharpness(imga), canny_sharpness(imga)]
        if stats[0] < 0.203:
            if stats[1] < 0.172:
                if stats[2] < 0.434:
                    # Class 1
                else:
                    # Class 2
            else:
                if stats[2] < 0.434:
                    # Class 3
                else:
                    # Class 4
        if stats[0] > 0.203 and stats[0] < 0.382:
            if stats[1] < 0.172:
                if stats[2] < 0.434:
                    # Class 5
                else:
                    # Class 6
            else:
                if stats[2] < 0.434:
                    # Class 7
                else:
                    # Class 8
        if stats[0] > 0.382:
            if stats[1] < 0.172:
                if stats[2] < 0.434:
                    # Class 9
                else:
                    # Class 10
            else:
                if stats[2] < 0.434:
                    # Class 11
                else:
                    # Class 12
        

