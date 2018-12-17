# -*- coding:utf-8 -*-
# -*- coding: utf-8 -*-
###############################################
#created by :  lxy
#Time:  2018/12/10 15:09
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

import tensorflow as tf
import tensorflow.contrib as tfc
import os, sys
import numpy as np
import time
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
import config as cfgs
sys.path.append(os.path.join(os.path.dirname(__file__),'../ssh'))
import build_ssh_net
sys.path.append(os.path.join(os.path.dirname(__file__),"../prepare_data"))
from read_tfrecord import Read_Tfrecord
sys.path.append(os.path.join(os.path.dirname(__file__),'../utils'))
import show_box_in_tensor
from get_property import load_property

def parms():
    parser = argparse.ArgumentParser(description='SSH training')
    parser.add_argument('--load-num',dest='load_num',type=int,default=0,help='ckpt num')
    parser.add_argument('--save-weight-period',dest='save_weight_period',type=int,default=5,\
                        help='the period to save')
    parser.add_argument('--epochs',type=int,default=20000,help='train epoch nums')
    parser.add_argument('--batch-size',dest='batch_size',type=int,default=1,\
                        help='train batch size')
    parser.add_argument('--model-path',dest='model_path',type=str,default='../../models/ssh',\
                        help='path saved models')
    parser.add_argument('--log-path',dest='log_path',type=str,default='../../logs',\
                        help='path saved logs')
    parser.add_argument('--gpu-list',dest='gpu_list',type=str,default='0',\
                        help='train on gpu num')
    parser.add_argument('--property-file',dest='property_file',type=str,\
                        default='../../data/property.txt',help='nums of train dataset images')
    parser.add_argument('--data-record-dir',dest='data_record_dir',type=str,\
                        default='../../data/',help='tensorflow data record')
    return parser.parse_args()

def train(args):
    '''
    model_path = kargs.get('model_path','../../models/ssh')
    load_num = kargs.get('load_num',0)
    log_dir = kargs.get('log_path','../../logs')
    epochs = kargs.get('epochs',20)
    batch_size = kargs.get('batch_size',1)
    save_weight_period = kargs.get('save_weight_period',1)
    property_file = kargs.get('property_file','../../data/property.txt')
    data_record = kargs.get('data_record','../../data/tfrecord')
    '''
    model_path = args.model_path
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    load_num = args.load_num
    log_dir = args.log_path
    epochs = args.epochs
    batch_size = args.batch_size
    save_weight_period = args.save_weight_period
    data_record_dir = args.data_record_dir
    property_file = os.path.join(data_record_dir,cfgs.DATASET_NAME,'property.txt')
    Property = load_property(property_file)
    train_img_nums = Property['img_nums']
    faster_rcnn = build_ssh_net.DetectionNetwork(base_network_name=cfgs.NET_NAME,
                                                       is_training=True)
    with tf.name_scope('get_batch'):
        tfrd = Read_Tfrecord(cfgs.DATASET_NAME,data_record_dir,batch_size,True)
        img_name_batch, img_batch, gtboxes_and_label_batch, num_obs_batch = tfrd.next_batch()
        #gtboxes_and_label = tf.reshape(gtboxes_and_label_batch, [-1, 5])
    # list as many types of layers as possible, even if they are not used now
    with tf.name_scope('build_ssh_trainnet'):
        result_dict, losses_dict = faster_rcnn.build_ssh_network(input_img_batch=img_batch,
                                            gtboxes_batch=gtboxes_and_label_batch,w_decay=cfgs.WEIGHT_DECAY)
    # ----------------------------------------------------------------------------------------------------build loss
    weight_decay_loss = tf.add_n(tf.losses.get_regularization_losses())
    # weight_decay_loss = tf.add_n(tf.losses.get_regularization_losses())
    bbox_loss_m1 = losses_dict['bbox_loss_m1']
    cls_loss_m1 = losses_dict['cls_loss_m1']
    total_loss_m1 = bbox_loss_m1 + cls_loss_m1

    bbox_loss_m2 = losses_dict['bbox_loss_m2']
    cls_loss_m2 = losses_dict['cls_loss_m2']
    total_loss_m2 = bbox_loss_m2 + cls_loss_m2

    bbox_loss_m3 = losses_dict['bbox_loss_m3']
    cls_loss_m3 = losses_dict['cls_loss_m3']
    total_loss_m3 = bbox_loss_m3 + cls_loss_m3

    total_loss = total_loss_m1 + total_loss_m2 + total_loss_m3 + weight_decay_loss

    # ---------------------------------------------------------------------------------------------------add summary
    tf.summary.scalar('SSH_M1_LOSS/cls_loss_m1', cls_loss_m1)
    tf.summary.scalar('SSH_M1_LOSS/bbox_loss_m1', bbox_loss_m1)
    tf.summary.scalar('SSH_M1_LOSS/total_loss_m1', total_loss_m1)

    tf.summary.scalar('SSH_M2_LOSS/cls_loss_m2', cls_loss_m2)
    tf.summary.scalar('SSH_M2_LOSS/bbox_loss_m2', bbox_loss_m2)
    tf.summary.scalar('SSH_M2_LOSS/total_loss_m2', total_loss_m2)

    tf.summary.scalar('SSH_M3_LOSS/cls_loss_m3', cls_loss_m3)
    tf.summary.scalar('SSH_M3_LOSS/bbox_loss_m3', bbox_loss_m3)
    tf.summary.scalar('SSH_M3_LOSS/total_loss_m3', total_loss_m3)

    tf.summary.scalar('LOSS/total_loss', total_loss)
    tf.summary.scalar('LOSS/regular_weights', weight_decay_loss)

    #gtboxes_in_img = show_box_in_tensor.draw_boxes_with_categories(img_batch=img_batch,
     #                                                              boxes=gtboxes_and_label[:, :-1],
      #                                                             labels=gtboxes_and_label[:, -1])
    if cfgs.ADD_BOX_IN_TENSORBOARD:
        detections_in_img_m1 = \
            show_box_in_tensor.draw_boxes_with_categories_and_scores(img_batch=img_batch,
                                                                     boxes=result_dict['final_bbox_m1'],
                                                                     labels=result_dict['final_category_m1'],
                                                                     scores=result_dict['final_scores_m1'])
        tf.summary.image('Compare/final_detection_m1', detections_in_img_m1)

        detections_in_img_m2 = \
            show_box_in_tensor.draw_boxes_with_categories_and_scores(img_batch=img_batch,
                                                                     boxes=result_dict['final_bbox_m2'],
                                                                     labels=result_dict['final_category_m2'],
                                                                     scores=result_dict['final_scores_m2'])
        tf.summary.image('Compare/final_detection_m2', detections_in_img_m2)

        detections_in_img_m3 = \
            show_box_in_tensor.draw_boxes_with_categories_and_scores(img_batch=img_batch,
                                                                     boxes=result_dict['final_bbox_m3'],
                                                                     labels=result_dict['final_category_m3'],
                                                                     scores=result_dict['final_scores_m3'])
        tf.summary.image('Compare/final_detection_m3', detections_in_img_m3)

    #tf.summary.image('Compare/gtboxes', gtboxes_in_img)

    global_step = tf.train.get_or_create_global_step()
    lr = tf.train.piecewise_constant(global_step,
                                     boundaries=[np.int64(cfgs.DECAY_STEP[0]), np.int64(cfgs.DECAY_STEP[1])],
                                     values=[cfgs.LR, cfgs.LR / 10., cfgs.LR / 100.])
    tf.summary.scalar('lr', lr)
    optimizer = tf.train.MomentumOptimizer(lr, momentum=cfgs.MOMENTUM)
    # ---------------------------------------------------------------------------------------------compute gradients
    gradients = optimizer.compute_gradients(total_loss)
    # enlarge_gradients for bias
    if cfgs.MUTILPY_BIAS_GRADIENT:
        gradients = faster_rcnn.enlarge_gradients_for_bias(gradients)
    if cfgs.GRADIENT_CLIPPING_BY_NORM:
        with tf.name_scope('clip_gradients'):
            gradients = tfc.training.clip_gradient_norms(gradients,
                                                          cfgs.GRADIENT_CLIPPING_BY_NORM)
    # train_op
    train_op = optimizer.apply_gradients(grads_and_vars=gradients,
                                         global_step=global_step)
    summary_op = tf.summary.merge_all()
    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )
    saver = tf.train.Saver(max_to_keep=30)
    config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(init_op)
        if load_num >0:
            model_path = "%s-%s" %(model_path,str(load_num) )
            model_dict = '/'.join(model_path.split('/')[:-1])
            ckpt = tf.train.get_checkpoint_state(model_dict)
            print("restore model path:",model_path)
            readstate = ckpt and ckpt.model_checkpoint_path
            assert readstate, "the params dictionary is not valid"
            saver.restore(sess, model_path)
            print("restore models' param")
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        summary_path = os.path.join(log_dir,'summary')
        if not os.path.exists(summary_path):
            os.makedirs(summary_path)
        summary_writer = tf.summary.FileWriter(summary_path, graph=sess.graph)
        try:
            for epoch_tmp in range(epochs):
                for step in range(np.ceil(train_img_nums/batch_size).astype(np.int32)):
                    training_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                    if step % cfgs.SHOW_TRAIN_INFO_INTE != 0 and step % cfgs.SMRY_ITER != 0:
                        _, global_stepnp = sess.run([train_op, global_step])
                    else:
                        if step % cfgs.SHOW_TRAIN_INFO_INTE == 0 and step % cfgs.SMRY_ITER != 0:
                            start = time.time()
                            _, global_stepnp, totalLoss,cls_m1,cls_m2,cls_m3,bbox_m1,bbox_m2,bbox_m3 = sess.run(
                                    [train_op, global_step, total_loss,cls_loss_m1,cls_loss_m2,cls_loss_m3,bbox_loss_m1, \
                                  bbox_loss_m2,bbox_loss_m3])
                            end = time.time()
                            print(""" {}: epoch{} step{}    |\t per_cost_time:{}s |\t total_loss:{} |\t cls_m1:{} |\t cls_m2:{}|\t cls_m3:{} |\t \
                                    bb_m1:{} |\t bb_m2:{} |\t bb_m3:{}  """ \
                                .format(training_time, epoch_tmp,global_stepnp,  (end - start),totalLoss,cls_m1,cls_m2,cls_m3,bbox_m1, \
                                  bbox_m2,bbox_m3 ))
                        else:
                            if step % cfgs.SMRY_ITER == 0:
                                _, global_stepnp, summary_str = sess.run([train_op, global_step, summary_op])
                                summary_writer.add_summary(summary_str, global_stepnp)
                                summary_writer.flush()
                if (epoch_tmp > 0 and epoch_tmp % save_weight_period == 0) or (epoch_tmp == epochs - 1):
                    save_dir = model_path
                    saver.save(sess, save_dir,global_step)
                    print(' weights had been saved')
        except tf.errors.OutOfRangeError:
            print("Trianing is over")
        finally:
            coord.request_stop()
            summary_writer.close()
        coord.join(threads)
        #record_file_out.close()
        sess.close()

if __name__ == '__main__':
    args = parms()
    gpu_group = args.gpu_list
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_group
    train(args)




