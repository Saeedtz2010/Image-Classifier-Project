import sys
import time 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from PIL import Image
import argparse
import json




def normalize(img):
    img = tf.cast(img, tf.float32)
    img /= 255
    return img
  
def process_image(img):
  img=tf.convert_to_tensor(img)
  img=tf.image.resize(img,[224,224])
  img=normalize(img)
  img=img.numpy()
  return img
    

def predict(image_path, model, top_k=5):
    
    image = Image.open(image_path)
    image = np.asarray(image)
    image = np.expand_dims(image,  axis=0)
    image = process_image(image)
    prob_list = model.predict(image)
    
def predict(image_path, model, k):
  im = Image.open(image_path)
  im=np.asarray(im)
  im=process_image(im)
  img = np.expand_dims(im, axis=0)
  out=model.predict(img)
  probs, classes = tf.math.top_k(out, k=k, sorted=True)
  return probs.numpy(), classes.numpy()


if __name__ == '__main__':
    print('predict.py, running')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('img_path')
    parser.add_argument('model_',default='my_model.h5')
    parser.add_argument('--top_k',default=5)
    parser.add_argument('--category_names',default='label_map.json') 
    
    
    args = parser.parse_args()
    print(args)
    
    print('image path:', args.img_path)
    print('model:', args.model_)
    print('top_k:', args.top_k)
    print('category_names:', args.category_names)
    
    image_path = args.img_path
    
    model = tf.keras.models.load_model(args.model_ ,custom_objects={'KerasLayer':hub.KerasLayer} )
    top_k = args.top_k

    with open(args.category_names, 'r') as f:
        class_names = json.load(f)
   
    probs, classes = predict(image_path, model, top_k)
    
    print(probs)
    print(classes)