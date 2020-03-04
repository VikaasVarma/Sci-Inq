import numpy as np
import pandas as pd
import cv2, os, json
from tqdm import tqdm

class Data(object):

    def __init__(self, path = "C:\\Users\\myeditha20\\Documents"):
        self.labels = {'car' : 1,
                       'person' : 2,
                       'rider' : 3,
                       'bus' : 4,
                       'bike' : 5,
                       'motor' : 6,
                       'truck' : 7,
                       'traffic light' : 8,
                       'train' : 9,
                       'traffic sign' : 10,
                      }
        self.path = path
        self.x = np.array(int)
        self.y = np.array(int)
        self.shape = (720, 1280)

        try:
            os.chdir(self.path)
        except FileNotFoundError:
            print('Invalid path to Data Files')

    def get_data(self, path, n = -1):
        with open('.\\Data\\' + path + '.json') as js:
            print("Loading json ...")
            d = json.load(js)
            if n == -1: n = len(d)
            self.x = np.zeros((n, *self.shape, 3))
            self.y = np.zeros((n, *self.shape))
            for i in tqdm(range(n)):
                self.x[i] = cv2.imread(".\\Data\\" + path + "\\" + d[i]['name'])
                y = np.zeros(self.shape)
                for j in d[i]['labels']:
                    if j['category'] in self.labels.keys():
                        bounds = dict(map(lambda kv: (kv[0], int(kv[1])), j['box2d'].items()))
                        y[bounds['y1']:bounds['y2'], bounds['x1']:bounds['x2']] = self.labels[j['category']]
                self.y[i] = y

    def get_train(self, n = -1):
        if (self.y and len(self.y) > n):
            return
        self.get_data("train", n)
    
    def get_val(self, n = -1):
        if (self.y and len(self.y) > n):
            return
        self.get_data("val", n)