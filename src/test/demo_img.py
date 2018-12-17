# -*- coding: utf-8 -*-
###############################################
#created by :  lxy
#Time:  2018/12/5 17:09
#project: Face detect
#company: 
#rversion: 0.1
#tool:   python 2.7
#modified:
#description  face detect 
####################################################

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import tensorflow as tf
import time
import cv2
import argparse
import numpy as np
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../prepare_data'))
from image_preprocess import short_side_resize_for_inference_data,norm_data,de_norm_data
sys.path.append(os.path.join(os.path.dirname(__file__), '../configs'))
import config as cfgs
sys.path.append(os.path.join(os.path.dirname(__file__), '../ssh'))
import build_ssh_net
sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))
import draw_box_in_img
import nms


def merge_result(result_dict):
    final_bbox = [result_dict['final_bbox_m1'], result_dict['final_bbox_m2'], result_dict['final_bbox_m2']]
    final_scores = [result_dict['final_scores_m1'], result_dict['final_scores_m2'], result_dict['final_scores_m3']]
    final_category = [result_dict['final_category_m1'], result_dict['final_category_m2'],
                      result_dict['final_category_m3']]

    final_bbox = np.concatenate(final_bbox, axis=0)
    final_scores = np.concatenate(final_scores, axis=0)
    final_category = np.concatenate(final_category, axis=0)

    keep = nms.bb_nms(final_bbox, final_scores, cfgs.FINAL_NMS_IOU_THRESHOLD)

    final_bbox = final_bbox[keep]
    final_scores = final_scores[keep]
    final_category = final_category[keep]

    return final_bbox, final_scores, final_category


def detect(det_net,inference_save_path,img_path,model_path):

    # 1. preprocess img
    img_plac = tf.placeholder(dtype=tf.uint8, shape=[None, None, 3])  # is RGB. not GBR
    img_batch = tf.cast(img_plac, tf.float32)
    #img_batch = short_side_resize_for_inference_data(img_tensor=img_batch,
     #                                                target_shortside_len=cfgs.IMG_SHORT_SIDE_LEN,
      #                                               length_limitation=cfgs.IMG_MAX_LENGTH)
    img_batch = tf.py_func(norm_data,[img_batch],tf.float32)
    img_batch.set_shape([None,None,3])
    img_batch = tf.expand_dims(img_batch, axis=0)  # [1, None, None, 3]
    result_dict = det_net.build_ssh_network(input_img_batch=img_batch,
                                            gtboxes_batch=None)
    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )
    restorer = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(init_op)
        restorer.restore(sess, model_path)
        print('restore model')
        raw_img = cv2.imread(img_path)
        start = time.time()
        resized_img, result_dict_ = \
        sess.run(
                [img_batch, result_dict],
                feed_dict={img_plac: raw_img[:, :, ::-1]}  # cv is BGR. But need RGB
                )
        end = time.time()
        detected_boxes, detected_scores, detected_categories = merge_result(result_dict_)
        # print("{} cost time : {} ".format(img_name, (end - start)))
        show_indices = detected_scores >= cfgs.SHOW_SCORE_THRSHOLD
        show_scores = detected_scores[show_indices]
        show_boxes = detected_boxes[show_indices]
        show_categories = detected_categories[show_indices]
        final_detections = draw_box_in_img.draw_boxes_with_label_and_scores(raw_img[:,:,::-1], #np.squeeze(resized_img, 0),
                                                                                boxes=show_boxes,
                                                                                labels=show_categories,
                                                                                scores=show_scores)
        #final_detections = de_norm_data(final_detections)
        #nake_name = a_img_name.split('/')[-1]
        # print (inference_save_path + '/' + nake_name)
        cv2.imwrite(inference_save_path + '/' + 'det_test.jpg',
                    final_detections[:, :, ::-1])
        print('1 image cost {}s'.format( end - start))
        cv2.imshow('test',final_detections[:, :, ::-1])
        cv2.waitKey(0)


def inference(img_path, inference_save_path,model_path):
    faster_rcnn = build_ssh_net.DetectionNetwork(base_network_name=cfgs.NET_NAME,
                                                       is_training=False)
    detect(faster_rcnn, inference_save_path,img_path,model_path)


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='TestImgs...U need provide the test dir')
    parser.add_argument('--img-dir', dest='img_dir',
                        help='data path',
                        default='demos', type=str)
    parser.add_argument('--save-dir', dest='save_dir',
                        help='demo imgs to save',
                        default='inference_results', type=str)
    parser.add_argument('--model-dir', dest='model_dir',
                        help='models',
                        default='inference_results', type=str)
    parser.add_argument('--GPU', dest='GPU',
                        help='gpu id ',
                        default='0', type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
    inference(args.img_dir,
              args.save_dir,
              args.model_dir)
















