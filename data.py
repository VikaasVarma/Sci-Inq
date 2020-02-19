import numpy as np
import cv2
import os
import json
from tqdm import tqdm

class Data(object):

    def __init__(self, path = "C:\\Users\\myeditha20\\Documents\\Yeet"):
        self.path = path
        self.train = ([],[])
        self.val = ([],[])
        self.data = {'train' : self.train, 'val' : self.val}

        try:
            os.chdir(self.path)
        except FileNotFoundError:
            print('Invalid path')

    def get_data(self, path, n = -1):
        self.data[path][0].clear()
        self.data[path][1].clear()
        with open('.\\Data\\' + path + '.json') as js:
            d = json.load(js)
            if n == -1: n = len(d)
            for i in tqdm(range(n)):
                self.data[path][0].append(cv2.imread(".\\Data\\" + path + "\\" + d[i]['name']))
                self.data[path][1].append([])
                for j in d[i]['labels']:
                    if 'box2d' not in j.keys():
                        continue
                    y = {'label' : [], 'box' : []}
                    y['label'].append(j['category'])
                    y['box'].append(j['box2d'])
                    self.data[path][1][-1].append(y)

    def get_train(self, n = -1):
        self.get_data("train", n)
        return self.train
    
    def get_val(self, n = -1):
        self.get_data("val", n)
        return self.val