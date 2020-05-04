import numpy as np
import pickle
import cv2
from os import listdir
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.models import load_model

default_image_size=tuple((256,256))
def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        plt.imshow(image)
        plt.show()
        if image is not None :
            image = cv2.resize(image, default_image_size)   
            return img_to_array(image)
        else :
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None

with open('C:/Users/devesh/Desktop/cnn_model.pkl','rb') as f:
    model=pickle.load(f)
with open('C:/Users/devesh/Desktop/label_transform.pkl','rb') as f1:
    lavel=pickle.load(f1)

EPOCHS=25
INIT_LR=1e-3

opt=Adam(lr=INIT_LR,decay=INIT_LR/EPOCHS)

model.compile(loss='binary_crossentropy',optimizer=opt, metrics=['accuracy'])
img=convert_image_to_array('C:/Users/devesh/Desktop/imp.JPG')

imglist=list()
imglist.append(img)
np_image_list = np.array(imglist, dtype=np.float16) / 225.0
classes=model.predict_classes(np_image_list)
result=lavel.classes_[classes[0]]
if(result[0]=='T'):
    a='Tomato'
else:
    a='Pomato'
print('Name of Plant',a)
print('Name of Disease',result)
