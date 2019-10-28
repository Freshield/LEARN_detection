#coding=utf-8
"""
@Author: Freshield
@License: (C) Copyright 2018, BEIJING LINKING MEDICAL TECHNOLOGY CO., LTD.
@Contact: yangyufresh@163.com
@File: a1_make_dataset.py
@Time: 2019-10-27 21:01
@Last_update: 2019-10-27 21:01
@Desc: None
@==============================================@
@      _____             _   _     _   _       @
@     |   __|___ ___ ___| |_|_|___| |_| |      @
@     |   __|  _| -_|_ -|   | | -_| | . |      @
@     |__|  |_| |___|___|_|_|_|___|_|___|      @
@                                    Freshield @
@==============================================@
"""
import os,cv2,sys,shutil

from xml.dom.minidom import Document

def writexml(filename,saveimg,bboxes,xmlpath):
    doc = Document()

    annotation = doc.createElement('annotation')

    doc.appendChild(annotation)

    folder = doc.createElement('folder')

    folder_name = doc.createTextNode('widerface')
    folder.appendChild(folder_name)
    annotation.appendChild(folder)
    filenamenode = doc.createElement('filename')
    filename_name = doc.createTextNode(filename)
    filenamenode.appendChild(filename_name)
    annotation.appendChild(filenamenode)
    source = doc.createElement('source')
    annotation.appendChild(source)
    database = doc.createElement('database')
    database.appendChild(doc.createTextNode('wider face Database'))
    source.appendChild(database)
    annotation_s = doc.createElement('annotation')
    annotation_s.appendChild(doc.createTextNode('PASCAL VOC2007'))
    source.appendChild(annotation_s)
    image = doc.createElement('image')
    image.appendChild(doc.createTextNode('flickr'))
    source.appendChild(image)
    flickrid = doc.createElement('flickrid')
    flickrid.appendChild(doc.createTextNode('-1'))
    source.appendChild(flickrid)
    owner = doc.createElement('owner')
    annotation.appendChild(owner)
    flickrid_o = doc.createElement('flickrid')
    flickrid_o.appendChild(doc.createTextNode('yanyu'))
    owner.appendChild(flickrid_o)
    name_o = doc.createElement('name')
    name_o.appendChild(doc.createTextNode('yanyu'))
    owner.appendChild(name_o)

    size = doc.createElement('size')
    annotation.appendChild(size)

    width = doc.createElement('width')
    width.appendChild(doc.createTextNode(str(saveimg.shape[1])))
    height = doc.createElement('height')
    height.appendChild(doc.createTextNode(str(saveimg.shape[0])))
    depth = doc.createElement('depth')
    depth.appendChild(doc.createTextNode(str(saveimg.shape[2])))

    size.appendChild(width)

    size.appendChild(height)
    size.appendChild(depth)
    segmented = doc.createElement('segmented')
    segmented.appendChild(doc.createTextNode('0'))
    annotation.appendChild(segmented)
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        objects = doc.createElement('object')
        annotation.appendChild(objects)
        object_name = doc.createElement('name')
        object_name.appendChild(doc.createTextNode('face'))
        objects.appendChild(object_name)
        pose = doc.createElement('pose')
        pose.appendChild(doc.createTextNode('Unspecified'))
        objects.appendChild(pose)
        truncated = doc.createElement('truncated')
        truncated.appendChild(doc.createTextNode('1'))
        objects.appendChild(truncated)
        difficult = doc.createElement('difficult')
        difficult.appendChild(doc.createTextNode('0'))
        objects.appendChild(difficult)
        bndbox = doc.createElement('bndbox')
        objects.appendChild(bndbox)
        xmin = doc.createElement('xmin')
        xmin.appendChild(doc.createTextNode(str(bbox[0])))
        bndbox.appendChild(xmin)
        ymin = doc.createElement('ymin')
        ymin.appendChild(doc.createTextNode(str(bbox[1])))
        bndbox.appendChild(ymin)
        xmax = doc.createElement('xmax')
        xmax.appendChild(doc.createTextNode(str(bbox[0] + bbox[2])))
        bndbox.appendChild(xmax)
        ymax = doc.createElement('ymax')
        ymax.appendChild(doc.createTextNode(str(bbox[1] + bbox[3])))
        bndbox.appendChild(ymax)
    f = open(xmlpath, "w")
    f.write(doc.toprettyxml(indent=''))
    f.close()


rootdir = '/media/freshield/SSD_1T/LEARN_detection/example/data/widerface'


def convertimgset(img_set):
    imgdir = os.path.join(rootdir, 'WIDER_%s' % img_set, 'images')
    gtfilepath = os.path.join(rootdir, 'wider_face_split', 'wider_face_%s_bbx_gt.txt' % img_set)

    fwrite = open(os.path.join(rootdir, 'ImageSets', 'Main', '%s.txt' % img_set), 'w')

    index = 0

    with open(gtfilepath, 'r') as gtfiles:
        while(True):
            filename = gtfiles.readline()[:-1]
            if filename is None or filename == '':
                break
            imgpath = os.path.join(imgdir, filename)

            img = cv2.imread(imgpath)

            try:
                test = img.data
            except Exception as e:
                print(filename)
                print(imgpath)
                print(e)

            numbbox = int(gtfiles.readline())
            if numbbox == 0:
                line = gtfiles.readline()
                continue

            bboxes = []

            for i in range(numbbox):
                line = gtfiles.readline()
                lines = line.split(' ')
                lines = lines[:4]

                bbox = tuple(int(i) for i in lines)

                if bbox[2] < 40 or bbox[3] < 40:
                    continue

                bboxes.append(bbox)

            filename = filename.replace('/', '_')

            if len(bboxes) == 0:
                print('no face')
                continue

            cv2.imwrite(os.path.join(rootdir, 'JPEGImages', filename), img)

            fwrite.write(filename.split('.')[0] + '\n')

            xmlpath = os.path.join(rootdir, 'Annotations', '%s.xml' % filename.split('.')[0])

            writexml(filename, img, bboxes, xmlpath)

            print('successs number is ', index)
            index += 1

    fwrite.close()



if __name__ == '__main__':
    img_sets = ['train', 'val']
    for img_set in img_sets:
        convertimgset(img_set)

    move_root = os.path.join(rootdir, 'ImageSets', 'Main')
    shutil.move(os.path.join(move_root, 'train.txt'),
                os.path.join(move_root, 'trainval.txt'))
    shutil.move(os.path.join(move_root, 'val.txt'),
                os.path.join(move_root, 'test.txt'))