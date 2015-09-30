# -*- coding: utf-8 -*-
"""
This is a software bundle "CRF-RNN", which is published in a ICCV paper titled "Conditional Random Fields as Recurrent Neural Networks". This is implemented as part of the Caffe library, written in C++/C. The current version is maintained by:

Shuai Zheng : szheng@robots.ox.ac.uk Sadeep Jayasumana : sadeep@robots.ox.ac.uk Bernardino Romera Paredes :

Supervisor: Philip Torr : philip.torr@eng.ox.ac.uk

For more information about CRF-RNN please vist the project website http://crfasrnn.torr.vision.
"""

caffe_root = '../caffe-crfrnn/'
import sys
sys.path.insert(0,caffe_root+'python')

import os
import cPickle
import logging
import numpy as np
import pandas as pd
from PIL import Image as PILImage
#import Image
import cStringIO as StringIO
import caffe
import matplotlib.pyplot as plt


MODEL_FILE = 'TVG_CRFRNN_COCO_VOC.prototxt'
PRETRAINED = 'TVG_CRFRNN_COCO_VOC.caffemodel'
IMAGE_FILE = '2007_000032.jpg'


#caffe.set_mode_gpu()
net = caffe.Segmenter(MODEL_FILE, PRETRAINED)
input_image = 255*caffe.io.load_image(IMAGE_FILE)


width = input_image.shape[0]
height = input_image.shape[1]
maxDim = max(width,height)

image = PILImage.fromarray(np.uint8(input_image))
image = np.array(image)

pallete = [0,0,0,
            128,0,0,
            0,128,0,
            128,128,0,
            0,0,128,
            128,0,128,
            0,128,128,
            128,128,128,
            64,0,0,
            192,0,0,
            64,128,0,
            192,128,0,
            64,0,128,
            192,0,128,
            64,128,128,
            192,128,128,
            0,64,0,
            128,64,0,
            0,192,0,
            128,192,0,
            0,64,128,
            128,64,128,
            0,192,128,
            128,192,128,
            64,64,0,
            192,64,0,
            64,192,0,
            192,192,0]

mean_vec = np.array([103.939, 116.779, 123.68], dtype=np.float32)
reshaped_mean_vec = mean_vec.reshape(1,1,3);

# Rearrange channels to form BGR
im = image[:,:,::-1]
# Subtract mean
im = im - reshaped_mean_vec

# Pad as necessary
cur_h, cur_w, cur_c = im.shape
pad_h = 500 - cur_h
pad_w = 500 - cur_w
im = np.pad(im, pad_width=((0, pad_h), (0, pad_w), (0, 0)), mode = 'constant', constant_values = 0)
# Get predictions
segmentation = net.predict([im])
segmentation2 = segmentation[0:cur_h,0:cur_w]
output_im = PILImage.fromarray(segmentation2)
output_im.putpalette(pallete)


plt.imshow(output_im)
plt.savefig('output.png')
