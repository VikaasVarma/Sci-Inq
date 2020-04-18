import numpy as np
import pandas as pd

from data import Data
from visualize import *

d = Data()

for i in range(12):
    d.create_tf_example('image_classes', category = 'class{0}'.format(i))