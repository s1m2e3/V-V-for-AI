import numpy as np
import pandas as pd
import imagecorruptions
#https://github.com/bethgelab/imagecorruptions
from PIL import Image

class Detector:
    def __init__(self,file_train,file_test):
        
    #def enhance(self,batch):
    #to be defined whether or not to enhance

    def corrupt(self,corruption_name,severity):

        corrupted = corrupt(image, corruption_number=corruption_name, severity=severity)
        return corrupted

    def get_data(self,file):
       
        import cPickle
        with open(file, 'rb') as fo:
            dictionary = cPickle.load(fo)
        return dictionary
    
    def train(self,batch):
    
    def test(self,batch):