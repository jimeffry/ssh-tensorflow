# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import xml.etree.cElementTree as ET
import numpy as np
import tensorflow as tf
import glob
import cv2
from image_preprocess import transform
import argparse
import os 
import sys
import math
sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
from label_dict import NAME_LABEL_MAP

def parms():
    parser = argparse.ArgumentParser(description='dataset convert')
    parser.add_argument('--VOC-dir',dest='VOC_dir',type=str,default='../../data/',\
                        help='dataset root')
    parser.add_argument('--xml-dir',dest='xml_dir',type=str,default='VOC_XML',\
                        help='xml files dir')
    parser.add_argument('--image-dir',dest='image_dir',type=str,default='VOC_JPG',\
                        help='images saved dir')
    parser.add_argument('--save-dir',dest='save_dir',type=str,default='../../data/',\
                        help='tfrecord save dir')
    parser.add_argument('--save-name',dest='save_name',type=str,\
                        default='train',help='image for train or test')
    parser.add_argument('--img-format',dest='img_format',type=str,\
                        default='.jpg',help='image format')
    parser.add_argument('--dataset-name',dest='dataset_name',type=str,default='VOC',\
                        help='datasetname')
    #for widerface
    parser.add_argument('--anno-file',dest='anno_file',type=str,\
                        default='../../data/wider_gt.txt',help='annotation files')
    parser.add_argument('--property-file',dest='property_file',type=str,\
                        default='../../data/property.txt',help='datasetname')
    return parser.parse_args()

class DataToRecord(object):
    def __init__(self,save_path):
        self.writer = tf.python_io.TFRecordWriter(path=save_path)

    def _int64_feature(self,value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(self,value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
    def write_recore(self,img_dict):
        # maybe do not need encode() in linux
        img_name = img_dict['img_name']
        img_height,img_width = img_dict['img_shape']
        img = img_dict['img_data']
        gtbox_label = img_dict['gt']
        feature = tf.train.Features(feature={
            'img_name': self._bytes_feature(img_name),
            'img_height': self._int64_feature(img_height),
            'img_width': self._int64_feature(img_width),
            'img': self._bytes_feature(img.tostring()),
            'gtboxes_and_label': self._bytes_feature(gtbox_label.tostring()),
            'num_objects': self._int64_feature(gtbox_label.shape[0])
        })
        example = tf.train.Example(features=feature)
        self.writer.write(example.SerializeToString())
    def close(self):
        self.writer.close()

# convert pascal voc data to tfrecord
def read_xml_gtbox_and_label(xml_path):
    """
    :param xml_path: the path of voc xml
    :return: a list contains gtboxes and labels, shape is [num_of_gtboxes, 5],
           and has [xmin, ymin, xmax, ymax, label] in a per row
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    img_width = None
    img_height = None
    box_list = []
    for child_of_root in root:
        # if child_of_root.tag == 'filename':
        #     assert child_of_root.text == xml_path.split('/')[-1].split('.')[0] \
        #                                  + FLAGS.img_format, 'xml_name and img_name cannot match'
        if child_of_root.tag == 'size':
            for child_item in child_of_root:
                if child_item.tag == 'width':
                    img_width = int(child_item.text)
                if child_item.tag == 'height':
                    img_height = int(child_item.text)
        if child_of_root.tag == 'object':
            label = None
            for child_item in child_of_root:
                if child_item.tag == 'name':
                    if child_item.text in NAME_LABEL_MAP.keys():
                        label = NAME_LABEL_MAP[child_item.text]
                    else:
                        continue
                if child_item.tag == 'bndbox':
                    for node in child_item:
                        if node.tag == 'xmin':
                            x1 = int(node.text)
                        if node.tag == 'ymin':
                            y1 = int(node.text)
                        if node.tag == 'xmax':
                            x2= int(node.text)
                        if node.tag == 'ymax':
                            y2 = int(node.text)
                    tmp_box=[x1,y1,x2,y2]  # [x1, y1. x2, y2]
                    assert label is not None, 'label is none, error'
                    tmp_box.append(label)  # [x1, y1. x2, y2, label]
                    box_list.append(tmp_box)
    if len(box_list) >0:
        gtbox_label = np.array(box_list, dtype=np.int32)  # [x1, y1. x2, y2, label]
    else:
        gtbox_label = None
    return img_height, img_width, gtbox_label

def convert_pascal_to_tfrecord(args):
    '''
    xml_dir = kargs.get('xml_dir',None)
    voc_dir = kargs.get('VOC_dir',None)
    save_dir = kargs.get('save_dir',None)
    dataset_name = kargs.get('dataset_name',None)
    image_dir = kargs.get('image_dir',None)
    save_name = kargs.get('save_name',None)
    img_format = kargs.get('img_format',None)
    '''
    xml_dir = args.xml_dir
    voc_dir = args.VOC_dir
    save_dir = args.save_dir
    dataset_name = args.dataset_name
    image_dir = args.image_dir
    save_name = args.save_name
    img_format = args.img_format
    #property_file = kargs.get('property_file',None)
    xml_path = os.path.join(voc_dir,xml_dir)
    image_path = os.path.join(voc_dir,image_dir)
    save_path = os.path.join(save_dir,dataset_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_name = save_name + '.tfrecord'
    record_save_path = os.path.join(save_path,save_name)
    # writer_options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
    record_w = DataToRecord(record_save_path)
    property_file = os.path.join(save_path,'property.txt')
    property_w = open(property_file,'w')
    total_img = 0
    dataset_img_num = len(glob.glob(xml_path + '/*.xml'))
    for count, xml in enumerate(glob.glob(xml_path + '/*.xml')):
        # to avoid path error in different development platform
        img_dict = dict()
        xml = xml.replace('\\', '/')
        img_name = xml.split('/')[-1].split('.')[0] + img_format
        img_path = os.path.join(img_path,img_name)
        if not os.path.exists(img_path):
            print('{} is not exist!'.format(img_path))
            continue
        img_height, img_width, gtbox_label = read_xml_gtbox_and_label(xml)
        # img = np.array(Image.open(img_path))
        if gtbox_label is None:
            continue
        img = cv2.imread(img_path)
        if img is None:
            print('{} is not None!'.format(img_path))
            continue
        #bgr to rgb
        img = img[:,:,::-1]
        img_dict['img_data'] = img
        img_dict['img_shape'] = [img_height,img_width]
        img_dict['gt'] = gtbox_label
        img_dict['img_name'] = img_name
        record_w.write_recore(img_dict)
        total_img+=1
        view_bar('Conversion progress', count + 1,dataset_img_num)
    print('\nConversion is complete!')
    property_w.write("{},{}".format(len(NAME_LABEL_MAP.keys()),total_img))
    property_file.close()
    record_w.close()

#convert widerface data to tfrecord
def rd_anotation(annotation,im_dir,img_format):
    '''
    annotation: 1/img_01 x1 y1 x2 y2 x1 y1 x2 y2 ...
    '''
    img_dict = dict()
    annotation = annotation.strip().split()
    im_path = annotation[0]
    #boxed change to float type
    bbox = map(float, annotation[1:])
    #gt
    boxes = np.array(bbox, dtype=np.int32).reshape(-1, 4)
    label = np.ones([boxes.shape[0],1],dtype=np.int32)*NAME_LABEL_MAP['face']
    #print(boxes.shape,label.shape)
    gt_box_labels = np.concatenate((boxes,label),axis=1)
    #load image
    img_path = os.path.join(im_dir, im_path + img_format)
    if not os.path.exists(img_path):
        return None
    img = cv2.imread(img_path)
    if img is None:
        return None
    img_name = img_path.split('/')[-1]
    img_name = img_name + img_format
    img_dict['img_data'] = img
    img_dict['img_shape'] = img.shape[:2]
    img_dict['gt'] = gt_box_labels
    img_dict['img_name'] = img_name
    return img_dict


def convert_widerface_to_tfrecord(args):
    '''
    anno_file = kargs.get('anno_file',None)
    save_dir = kargs.get('save_dir',None)
    dataset_name = kargs.get('dataset_name',None)
    image_dir = kargs.get('image_dir',None)
    save_name = kargs.get('save_name',None)
    img_format = kargs.get('img_format',None)
    #property_file = kargs.get('property_file',None)
    '''
    anno_file = args.anno_file
    save_dir = args.save_dir
    dataset_name = args.dataset_name
    image_dir = args.image_dir
    save_name = args.save_name
    img_format = args.img_format
    save_path = os.path.join(save_dir,dataset_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_name = save_name + '.tfrecord'
    record_save_path = os.path.join(save_path,save_name)
    record_w = DataToRecord(record_save_path)
    property_file = os.path.join(save_path,'property.txt')
    property_w = open(property_file,'w')
    anno_p = open(anno_file,'r')
    anno_lines = anno_p.readlines()
    total_img = 0
    dataset_img_num = len(anno_lines)
    for count,tmp in enumerate(anno_lines):
        img_dict = rd_anotation(tmp,image_dir,img_format)
        if img_dict is None:
            print("the img path is none:",tmp.strip().split()[0])
            continue
        record_w.write_recore(img_dict)
        total_img+=1
        view_bar('Conversion progress', count + 1,dataset_img_num)
    print('\nConversion is complete!')
    property_w.write("{},{}".format(len(NAME_LABEL_MAP.keys()),total_img))
    property_file.close()
    record_w.close()
    anno_p.close()


def view_bar(message, num, total):
    rate = num / total
    rate_num = int(rate * 40)
    rate_nums = math.ceil(rate * 100)
    r = '\r%s:[%s%s]%d%%\t%d/%d' % (message, ">" * rate_num, " " * (40 - rate_num), rate_nums, num, total,)
    sys.stdout.write(r)
    sys.stdout.flush()

if __name__ == '__main__':
    args = parms()
    dataset = args.dataset_name
    if 'Wider' in dataset:
        convert_widerface_to_tfrecord(args)
    elif 'VOC' in dataset:
        convert_pascal_to_tfrecord(args)
