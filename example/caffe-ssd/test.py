import os
import cv2
import sys
import numpy as np

caffe_root = "/home/kuan/code/caffe-ssd"

os.chdir(caffe_root)

sys.path.insert(0,os.path.join(caffe_root, 'python'))

import caffe

caffe.set_device(0)
caffe.set_mode_gpu()

model_def = "/home/kuan/PycharmProjects/muke/caffe-ssd/deploy.prototxt"
model_weight = "/home/kuan/PycharmProjects/muke/caffe-ssd/face_model.caffemodel"

img_path = "/home/kuan/PycharmProjects/muke/caffe-ssd/30_Surgeons_Surgeons_30_90.jpg"

net = caffe.Net(model_def,model_weight,caffe.TEST)

image_data = caffe.io.load_image(img_path)

tranformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})

tranformer.set_transpose('data', (2,0,1))



tranformer.set_mean('data',np.array([104,117,123]))

tranformer.set_raw_scale('data', 255)


tranformer_image = tranformer.preprocess('data', image_data)

net.blobs['data'].reshape(1,3,300,300)

net.blobs['data'].data[...] = tranformer_image

detect_out = net.forward()['detection_out']

print detect_out

det_label = detect_out[0,0,:,1]
det_conf  = detect_out[0,0,:,2]

det_xmin = detect_out[0,0,:,3]
det_ymin = detect_out[0,0,:,4]
det_xmax = detect_out[0,0,:,5]
det_ymax = detect_out[0,0,:,6]

top_indices = [i for i , conf in enumerate(det_conf) if conf >=0.1]

top_conf = det_conf[top_indices]

top_xmin = det_xmin[top_indices]
top_ymin = det_ymin[top_indices]
top_xmax = det_xmax[top_indices]
top_ymax = det_ymax[top_indices]

[height,width,_] = image_data.shape

for i in range(min(5, top_conf.shape[0])):
    xmin = int(top_xmin[i] * width)
    ymin = int(top_ymin[i] * height)
    xmax = int(top_xmax[i] * width)
    ymax = int(top_ymax[i] * height)

    cv2.rectangle(image_data, (xmin,ymin),(xmax,ymax),(255,0,0),5)

cv2.imshow("face", image_data)

cv2.waitKey(0)
