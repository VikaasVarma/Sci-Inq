import numpy as np
import pandas as pd
import tensorflow as tf
from test.models.research.object_detection.utils import dataset_util
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
        self.bounds = []
        self.names = []
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
        self.labels = [[]] * (end - start)
        self.names = [''] * (end - start)
        bounds = {'x1' : 0, 'x2' : 0, 'y1' : 0, 'y2' : 0, 'category' : ''}
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
                    self.names[i - start] = value
                if prefix == "item.labels.item.category":
                    bounds['category'] = value
                if "box2d" in prefix and event == "number":
                    bounds[prefix[-2:]] = int(value)
                if prefix == "item.labels.item" and event == "end_map":
                    if bounds['category'] not in self.labels.keys():
                        continue
                    y[bounds['y1']:bounds['y2'], bounds['x1']:bounds['x2']] = self.labels[bounds['category']]
                    self.labels[i - start].append(bounds)
                if prefix == "item.labels" and event == "end_array":
                    self.y[i - start] = y

    def get_train(self, start = 0, end = 100):
        self.get_data("train", start, end)
    
    def get_val(self, start = 0, end = 100):
        self.get_data("val", start, end)

    def create_tf_example(path, cat = "temp"):
        flags = tf.app.flags
        flags.DEFINE_string('output_path', cat, 'Path to output TFRecord')
        FLAGS = flags.FLAGS

        os.chdir(path)
        width, height = 1280, 720
        filetype = b'jpg'

        data = pd.read_csv("data.csv")
        data.bounds = data.bounds.map(eval)
        tfexamples = []

        for i in data.values:
            filename = i[0]
            bounds = {'x1' : [], 'x2' : [], 'y1' : [], 'y2' : []}
            labels = []
            ids = []
            image = cv2.imread("..\\Data\\train\\" + filename)
            for j in i[1]:
                bounds['x1'].append(j['x1'] / width)
                bounds['x2'].append(j['x2'] / width)
                bounds['y1'].append(j['y1'] / height)
                bounds['y2'].append(j['y2'] / height)
                labels.append(j['category'])
                ids.append(self.labels(j['category']))

        tfexamples.append(tf.train.Example(features = tf.train.Features(feature = {
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(filename),
            'image/source_id': dataset_util.bytes_feature(filename),
            'image/encoded': dataset_util.bytes_feature(image),
            'image/format': dataset_util.bytes_feature(filetype),
            'image/object/bbox/xmin': dataset_util.float_list_feature(bounds['x1']),
            'image/object/bbox/xmax': dataset_util.float_list_feature(bounds['x2']),
            'image/object/bbox/ymin': dataset_util.float_list_feature(bounds['y1']),
            'image/object/bbox/ymax': dataset_util.float_list_feature(bounds['y2']),
            'image/object/class/text': dataset_util.bytes_list_feature(labels),
            'image/object/class/label': dataset_util.int64_list_feature(ids)
        })))
        
        writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

        for example in tfexamples:
            writer.write(example.SerializeToString())

        writer.close()