import pandas as pd
import numpy as np
from algos import *

class Classifier(object):
    def __init__(self):
        self.prop = pd.DataFrame(columns = ['id',
                                            'fourier',
                                            'canny',
                                            'brightness',
                                            'precipitation'
                                            ])

    def get_properties(self, x):
        for i, img in enumerate(x):
            entry = [i,
                     fourier_sharpness(x),
                     canny_sharpness(x),
                     brightness(x),
                     weather(x)
                    ]
            self.prop.loc[len(self.prop)] = entry
    
    def splitby(self):
        pass