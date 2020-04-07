import numpy as np
import pandas as pd
import cv2, os, json, ijson
from tqdm import tqdm

path = "D:\\se2\\Sci-Inq"

class Data(object):

    def __init__(self, path = path):
        self.labels = {
            '' : 0,
            'car' : 1,
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

    def get_data(self, path, start, end):
        parser = ijson.parse(open('.\\Data\\' + path + '.json'))
        i = 0
        self.x = np.zeros((end - start, *self.shape, 3))
        self.y = np.zeros((end - start, *self.shape))
        category = ''
        bounds = {'x1' : 0, 'x2' : 0, 'y1' : 0, 'y2' : 0}
        with tqdm(total = end - start) as pbar:
            for prefix, event, value in parser:
                if i == end:
                    break
                if prefix == "item" and event == "end_map":
                    i += 1
                    pbar.update(0 if i < start else 1)
                if i < start:
                    continue

                
                if prefix == "item.name" and event == "string":
                    self.x[i - start] = cv2.imread(".\\Data\\" + path + "\\" + value)
                    y = np.zeros(self.shape)
                if prefix == "item.labels.item.category":
                    category = value
                if "box2d" in prefix and event == "number":
                    bounds[prefix[-2:]] = int(value)
                if prefix == "item.labels.item" and event == "end_map":
                    if category not in self.labels.keys():
                        continue
                    y[bounds['y1']:bounds['y2'], bounds['x1']:bounds['x2']] = self.labels[category]
                if prefix == "item.labels" and event == "end_array":
                    self.y[i - start] = y

    def get_train(self, start = 0, end = 100):
        self.get_data("train", start, end)
    
    def get_val(self, start = 0, end = 100):
        self.get_data("val", start, end)