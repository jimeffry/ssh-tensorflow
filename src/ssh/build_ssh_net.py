# -*-coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import os
import tensorflow as tf
import tensorflow.contrib.layers as tfc
import numpy as np
from proposal_target_layer import proposal_target_layer
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../network'))
import resnet as models
import mobilenetV2 as MobileNetV2
sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))
import encode_and_decode
import boxes_utils
import anchor_utils
import show_box_in_tensor
sys.path.append(os.path.join(os.path.dirname(__file__), '../configs'))
import config as cfgs
sys.path.append(os.path.join(os.path.dirname(__file__), '../losses'))
import losses


class DetectionNetwork(object):
    def __init__(self, base_network_name, is_training):
        self.base_network_name = base_network_name
        self.is_training = is_training
        self.use_bn= cfgs.BN_USE
        self.w_regular = tfc.l2_regularizer(cfgs.WEIGHT_DECAY)
        self.m1_num_anchors_per_location = len(cfgs.M1_ANCHOR_SCALES) * len(cfgs.ANCHOR_RATIOS)
        self.m2_num_anchors_per_location = len(cfgs.M2_ANCHOR_SCALES) * len(cfgs.ANCHOR_RATIOS)
        self.m3_num_anchors_per_location = len(cfgs.M3_ANCHOR_SCALES) * len(cfgs.ANCHOR_RATIOS)

    def build_base_network(self, input_img_batch,**kargs):
        with tf.variable_scope("base_network"):
            if self.base_network_name.startswith('resnet'):
                return models.get_symble(input_img_batch, net_name=self.base_network_name,\
                                        train_fg=self.is_training,**kargs)
            elif self.base_network_name.startswith('mobilenet'):
                return MobileNetV2.get_symble(input_img_batch,net_name=self.base_network_name,\
                                         train_fg=self.is_training,**kargs)
            else:
                raise ValueError('Sry, we only support resnet or mobilenet_v2')

    def postprocess_ssh(self, rois, bbox_ppred, scores, img_shape, iou_threshold):
        '''
        :param rois:[-1, 4]
        :param bbox_ppred: [-1, (cfgs.Class_num+1) * 4]
        :param scores: [-1, cfgs.Class_num + 1]
        :return:
        '''
        with tf.variable_scope('postprocess_fastrcnn'):
            if self.is_training:
                pre_nms_topN = cfgs.TOP_K_NMS_TRAIN
                post_nms_topN = cfgs.MAXIMUM_PROPOSAL_TARIN
            else:
                pre_nms_topN = cfgs.TOP_K_NMS_TEST
                post_nms_topN = cfgs.MAXIMUM_PROPOSAL_TEST
            rois = tf.stop_gradient(rois)
            scores = tf.stop_gradient(scores)
            bbox_ppred = tf.reshape(bbox_ppred, [-1, cfgs.CLASS_NUM + 1, 4])
            bbox_ppred = tf.stop_gradient(bbox_ppred)
            bbox_pred_list = tf.unstack(bbox_ppred, axis=1)
            score_list = tf.unstack(scores, axis=1)
            allclasses_boxes = []
            allclasses_scores = []
            categories = []
            for i in range(1, cfgs.CLASS_NUM + 1):
                # 1. decode boxes in each class
                tmp_encoded_box = bbox_pred_list[i]
                tmp_score = score_list[i]
                tmp_decoded_boxes = encode_and_decode.decode_boxes(encoded_boxes=tmp_encoded_box,
                                                                   reference_boxes=rois,
                                                                   scale_factors=cfgs.ROI_SCALE_FACTORS)
                # 2. clip to img boundaries
                tmp_decoded_boxes = boxes_utils.clip_boxes_to_img_boundaries(decode_boxes=tmp_decoded_boxes,
                                                                             img_shape=img_shape)
                if pre_nms_topN > 0:
                    pre_nms_topN = tf.minimum(pre_nms_topN, tf.shape(tmp_decoded_boxes)[0])
                    tmp_score, top_k_indices = tf.nn.top_k(tmp_score, k=pre_nms_topN)
                    tmp_decoded_boxes = tf.gather(tmp_decoded_boxes, top_k_indices)
                # 3. NMS
                keep = tf.image.non_max_suppression(boxes=tmp_decoded_boxes,
                                                    scores=tmp_score,
                                                    max_output_size=cfgs.SSH_NMS_MAX_BOXES_PER_CLASS,
                                                    iou_threshold=iou_threshold)
                perclass_boxes = tf.gather(tmp_decoded_boxes, keep)
                perclass_scores = tf.gather(tmp_score, keep)
                allclasses_boxes.append(perclass_boxes)
                allclasses_scores.append(perclass_scores)
                categories.append(tf.ones_like(perclass_scores) * i)
            final_boxes = tf.concat(allclasses_boxes, axis=0)
            final_scores = tf.concat(allclasses_scores, axis=0)
            final_category = tf.concat(categories, axis=0)
            if self.is_training:
                '''
                in training. We should show the detecitons in the tensorboard. So we add this.
                '''
                kept_indices = tf.reshape(tf.where(tf.greater_equal(final_scores, cfgs.SHOW_SCORE_THRSHOLD)), [-1])
                final_boxes = tf.gather(final_boxes, kept_indices)
                final_scores = tf.gather(final_scores, kept_indices)
                final_category = tf.gather(final_category, kept_indices)
        return final_boxes, final_scores, final_category

    def context_module(self, feature_maps,channels):
        with tf.variable_scope('context_module'):
            #channels = feature_maps.get_shape().as_list()[-1]
            convbase = models.Conv_block(feature_maps,3,filter_num=channels //2,relu_type='relu6',\
                            bn_use=self.use_bn,train_fg=self.is_training,w_regular=self.w_regular,name='base_conv')
            conv1 = models.Conv_block(convbase,3,filter_num=channels //2,relu_type='relu6',\
                            bn_use=self.use_bn,train_fg=self.is_training,w_regular=self.w_regular,name='ct1_conv')
            conv2_1 = models.Conv_block(convbase,3,filter_num=channels //2,relu_type='relu6',\
                            bn_use=self.use_bn,train_fg=self.is_training,w_regular=self.w_regular,name='ct2_1_conv')
            conv2_2 = models.Conv_block(conv2_1,3,filter_num=channels //2,relu_type='relu6',\
                            bn_use=self.use_bn,train_fg=self.is_training,w_regular=self.w_regular,name='ct2_2_conv')
            output = tf.concat([conv1, conv2_2], axis=3, name='ct_concat')
            return output

    def detection_module(self, feature_maps, num_anchors_per_location,channels,scope):
        with tf.variable_scope(scope):
            #channels = feature_maps.get_shape().as_list()[-1]
            context = self.context_module(feature_maps,channels)
            convbase= models.Conv_block(feature_maps,3,filter_num=channels,relu_type='relu6',\
                                bn_use=self.use_bn,train_fg=self.is_training,w_regular=self.w_regular,name='dt_conv')
            concat_layer = tf.concat([convbase, context], axis=3, name='concat')
            cls_score = models.Conv_block(concat_layer,1,filter_num=num_anchors_per_location * (cfgs.CLASS_NUM + 1),\
                                bn_use=False,train_fg=self.is_training,w_regular=self.w_regular,name='cls_score')
            box_pred = models.Conv_block(concat_layer,1,filter_num=num_anchors_per_location * (cfgs.CLASS_NUM + 1) * 4,\
                                bn_use=False,train_fg=self.is_training,w_regular=self.w_regular,name='box_pred')
            return cls_score, box_pred

    def add_roi_batch_img_smry(self, img, rois, labels, name):
        positive_roi_indices = tf.reshape(tf.where(tf.greater_equal(labels, 1)), [-1])
        negative_roi_indices = tf.reshape(tf.where(tf.equal(labels, 0)), [-1])
        pos_roi = tf.gather(rois, positive_roi_indices)
        neg_roi = tf.gather(rois, negative_roi_indices)
        pos_in_img = show_box_in_tensor.only_draw_boxes(img_batch=img,
                                                        boxes=pos_roi)
        neg_in_img = show_box_in_tensor.only_draw_boxes(img_batch=img,
                                                        boxes=neg_roi)
        tf.summary.image('{}/pos_rois'.format(name), pos_in_img)
        tf.summary.image('{}/neg_rois'.format(name), neg_in_img)

    def get_detection_out(self,feature_map,anchors_per_location,channels,name_scope):
        with tf.variable_scope(name_scope):
            cls_score, box_pred = self.detection_module(feature_map,anchors_per_location,channels,name_scope)
            box_pred = tf.reshape(box_pred, [-1, 4 * (cfgs.CLASS_NUM + 1)])
            cls_score = tf.reshape(cls_score, [-1, (cfgs.CLASS_NUM + 1)])
            cls_prob = tfc.softmax(cls_score, scope='cls_prob')
            return box_pred,cls_score,cls_prob

    def get_feature_m1m3(self,feature_c4,feature_c5):
        with tf.variable_scope('get_feature_m1m3'):
            feature_m3 = tfc.max_pool2d(inputs=feature_c5,kernel_size=2,stride=2,scope='max_pool_ssh')
            channels_16 = feature_c5.get_shape().as_list()[-1]
            channels_8 = feature_c4.get_shape().as_list()[-1]
            feature8_shape = tf.shape(feature_c4)
            conv5_1= models.Conv_block(feature_c5,1,filter_num=channels_16 // 8,relu_type='relu6',\
                                bn_use=self.use_bn,train_fg=self.is_training,w_regular=self.w_regular,name='c5_conv')
            conv5_upsampling = tf.image.resize_bilinear(conv5_1, [feature8_shape[1], feature8_shape[2]],
                                                    name='m2_upsampling')
            conv4_1= models.Conv_block(feature_c4,1,filter_num=channels_8 // 4,relu_type='relu6',\
                                bn_use=self.use_bn,train_fg=self.is_training,w_regular=self.w_regular,name='c4_1_conv')
            eltwise_sum = tf.add(conv5_upsampling,conv4_1)
            feature_m1= models.Conv_block(eltwise_sum,3,filter_num=channels_8 // 4,relu_type='relu6',\
                                bn_use=self.use_bn,train_fg=self.is_training,w_regular=self.w_regular,\
                                name='feature_m1_conv')
            return feature_m1,feature_m3

    def get_unify_shape(self,rois,labels,bbox_targets):
        rois_m = tf.reshape(rois, [-1, 4],name='roi_shape')
        labels_m = tf.to_int32(labels)
        labels_m = tf.reshape(labels_m, [-1],name='label_shape')
        bbox_targets_m = tf.reshape(bbox_targets, [-1, 4 * (cfgs.CLASS_NUM + 1)],name='bbox_target_shape')
        return rois_m,labels_m,bbox_targets_m

    def get_pred_results_by_indx(self,box_pred,cls_score,cls_prob,keep_inds):
        box_pred_m = tf.gather(box_pred, keep_inds,name='box_pred_sel')
        cls_score_m = tf.gather(cls_score, keep_inds,name='cls_score_sel')
        cls_prob_m = tf.reshape(tf.gather(cls_prob, keep_inds,name='cls_prob_sel'), [-1, (cfgs.CLASS_NUM + 1)],name='cls_prob_reshape')
        return box_pred_m,cls_score_m,cls_prob_m

    def build_ssh_network(self, input_img_batch, gtboxes_batch,**kargs):
        if self.is_training:
            # ensure shape is [M, 5]
            gtboxes_batch = tf.reshape(gtboxes_batch, [-1, 5])
            gtboxes_batch = tf.cast(gtboxes_batch, tf.float32)
        img_shape = tf.shape(input_img_batch)
        # 1. build base network
        feature_stride8, feature_stride16 = self.build_base_network(input_img_batch,**kargs)
        # 2. build rpn
        with tf.variable_scope('build_ssh'):
            feature_m1,feature_m3 = self.get_feature_m1m3(feature_stride8,feature_stride16)
            box_pred_m3,cls_score_m3,cls_prob_m3 = self.get_detection_out(feature_m3,
                                                                        self.m3_num_anchors_per_location,
                                                                        cfgs.M3_CHANNELS,
                                                                        'detection_module_m3')
            box_pred_m2,cls_score_m2,cls_prob_m2 = self.get_detection_out(feature_stride16,
                                                                        self.m2_num_anchors_per_location,
                                                                        cfgs.M2_CHANNELS,
                                                                        'detection_module_m2')
            box_pred_m1,cls_score_m1,cls_prob_m1 = self.get_detection_out(feature_m1,
                                                                        self.m1_num_anchors_per_location,
                                                                        cfgs.M1_CHANNELS,
                                                                        'detection_module_m1')
        # 3. generate_anchors
        anchors_m1 = anchor_utils.make_anchors(base_anchor_size=cfgs.BASE_ANCHOR_SIZE_LIST[0],
                                               anchor_scales=cfgs.M1_ANCHOR_SCALES,
                                               anchor_ratios=cfgs.ANCHOR_RATIOS,
                                               featuremap=feature_stride8,
                                               stride=[cfgs.ANCHOR_STRIDE[0]],
                                               name="make_anchors_for_m1")
        anchors_m2 = anchor_utils.make_anchors(base_anchor_size=cfgs.BASE_ANCHOR_SIZE_LIST[0],
                                               anchor_scales=cfgs.M2_ANCHOR_SCALES,
                                               anchor_ratios=cfgs.ANCHOR_RATIOS,
                                               featuremap=feature_stride16,
                                               stride=[cfgs.ANCHOR_STRIDE[1]],
                                               name="make_anchors_for_m2")
        anchors_m3 = anchor_utils.make_anchors(base_anchor_size=cfgs.BASE_ANCHOR_SIZE_LIST[0],
                                               anchor_scales=cfgs.M3_ANCHOR_SCALES,
                                               anchor_ratios=cfgs.ANCHOR_RATIOS,
                                               featuremap=feature_m3,
                                               stride=[cfgs.ANCHOR_STRIDE[2]],
                                               name="make_anchors_for_m3")
        # refer to paper: Seeing Small Faces from Robust Anchor’s Perspective
        #***************get rois and gt_boxes for all classes
        if self.is_training:
            with tf.variable_scope('sample_ssh_minibatch_m1'):
                rois_m1, labels_m1, bbox_targets_m1, keep_inds_m1 = \
                    tf.py_func(proposal_target_layer,
                               [anchors_m1, gtboxes_batch, 'M1'],
                               [tf.float32, tf.float32, tf.float32, tf.int32],name='M1')
                rois_m1,labels_m1,bbox_targets_m1 = self.get_unify_shape(rois_m1,labels_m1,bbox_targets_m1)
                self.add_roi_batch_img_smry(input_img_batch, rois_m1, labels_m1, 'm1')
            with tf.variable_scope('sample_ssh_minibatch_m2'):
                rois_m2, labels_m2, bbox_targets_m2, keep_inds_m2 = \
                    tf.py_func(proposal_target_layer,
                               [anchors_m2, gtboxes_batch, 'M2'],
                               [tf.float32, tf.float32, tf.float32, tf.int32],name='M2')
                rois_m2,labels_m2,bbox_targets_m2 = self.get_unify_shape(rois_m2,labels_m2,bbox_targets_m2)
                self.add_roi_batch_img_smry(input_img_batch, rois_m2, labels_m2, 'm2')
            with tf.variable_scope('sample_ssh_minibatch_m3'):
                rois_m3, labels_m3, bbox_targets_m3, keep_inds_m3 = \
                    tf.py_func(proposal_target_layer,
                               [anchors_m3, gtboxes_batch, 'M3'],
                               [tf.float32, tf.float32, tf.float32, tf.int32],name='M3')
                rois_m3,labels_m3,bbox_targets_m3 = self.get_unify_shape(rois_m3,labels_m3,bbox_targets_m3)
                self.add_roi_batch_img_smry(input_img_batch, rois_m3, labels_m3, 'm3')

        if not self.is_training:
            with tf.variable_scope('postprocess_ssh_m1'):
                final_bbox_m1, final_scores_m1, final_category_m1 = self.postprocess_ssh(rois=anchors_m1,
                                                                                         bbox_ppred=box_pred_m1,
                                                                                         scores=cls_prob_m1,
                                                                                         img_shape=img_shape,
                                                                                         iou_threshold=cfgs.M1_NMS_IOU_THRESHOLD)

            with tf.variable_scope('postprocess_ssh_m2'):
                final_bbox_m2, final_scores_m2, final_category_m2 = self.postprocess_ssh(rois=anchors_m2,
                                                                                         bbox_ppred=box_pred_m2,
                                                                                         scores=cls_prob_m2,
                                                                                         img_shape=img_shape,
                                                                                         iou_threshold=cfgs.M2_NMS_IOU_THRESHOLD)

            with tf.variable_scope('postprocess_ssh_m3'):
                final_bbox_m3, final_scores_m3, final_category_m3 = self.postprocess_ssh(rois=anchors_m3,
                                                                                         bbox_ppred=box_pred_m3,
                                                                                         scores=cls_prob_m3,
                                                                                         img_shape=img_shape,
                                                                                         iou_threshold=cfgs.M3_NMS_IOU_THRESHOLD)

            result_dict = {'final_bbox_m1': final_bbox_m1, 'final_scores_m1': final_scores_m1,
                           'final_category_m1': final_category_m1, 'final_bbox_m2': final_bbox_m2,
                           'final_scores_m2': final_scores_m2, 'final_category_m2': final_category_m2,
                           'final_bbox_m3': final_bbox_m3, 'final_scores_m3': final_scores_m3,
                           'final_category_m3': final_category_m3}
            return result_dict

        else:
            with tf.variable_scope('ssh_loss_m1'):
                if not cfgs.M1_MINIBATCH_SIZE == -1:
                    box_pred_m1,cls_score_m1,cls_prob_m1 = self.get_pred_results_by_indx(box_pred_m1,cls_score_m1,cls_prob_m1,keep_inds_m1)
                    bbox_loss_m1 = losses.smooth_l1_loss_rcnn(bbox_pred=box_pred_m1,
                                                              bbox_targets=bbox_targets_m1,
                                                              label=labels_m1,
                                                              num_classes=cfgs.CLASS_NUM + 1,
                                                              sigma=cfgs.M1_SIGMA)
                    cls_loss_m1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score_m1,
                                                                                                labels=labels_m1))
            with tf.variable_scope('postprocess_ssh_m1'):
                final_bbox_m1, final_scores_m1, final_category_m1 = self.postprocess_ssh(rois=rois_m1,
                                                                                         bbox_ppred=box_pred_m1,
                                                                                         scores=cls_prob_m1,
                                                                                         img_shape=img_shape,
                                                                                         iou_threshold=cfgs.M1_NMS_IOU_THRESHOLD)
            with tf.variable_scope('ssh_loss_m2'):
                if not cfgs.M2_MINIBATCH_SIZE == -1:
                    box_pred_m2,cls_score_m2,cls_prob_m2 = self.get_pred_results_by_indx(box_pred_m2,cls_score_m2,cls_prob_m2,keep_inds_m2)
                    bbox_loss_m2 = losses.smooth_l1_loss_rcnn(bbox_pred=box_pred_m2,
                                                              bbox_targets=bbox_targets_m2,
                                                              label=labels_m2,
                                                              num_classes=cfgs.CLASS_NUM + 1,
                                                              sigma=cfgs.M2_SIGMA)
                    cls_loss_m2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score_m2,
                                                                                                labels=labels_m2))
            with tf.variable_scope('postprocess_ssh_m2'):
                final_bbox_m2, final_scores_m2, final_category_m2 = self.postprocess_ssh(rois=rois_m2,
                                                                                         bbox_ppred=box_pred_m2,
                                                                                         scores=cls_prob_m2,
                                                                                         img_shape=img_shape,
                                                                                         iou_threshold=cfgs.M2_NMS_IOU_THRESHOLD)
            with tf.variable_scope('ssh_loss_m3'):
                if not cfgs.M3_MINIBATCH_SIZE == -1:
                    box_pred_m3,cls_score_m3,cls_prob_m3 = self.get_pred_results_by_indx(box_pred_m3,cls_score_m3,cls_prob_m3,keep_inds_m3)
                    bbox_loss_m3 = losses.smooth_l1_loss_rcnn(bbox_pred=box_pred_m3,
                                                              bbox_targets=bbox_targets_m3,
                                                              label=labels_m3,
                                                              num_classes=cfgs.CLASS_NUM + 1,
                                                              sigma=cfgs.M3_SIGMA)
                    cls_loss_m3 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score_m3,
                                                                                                labels=labels_m3))
            with tf.variable_scope('postprocess_ssh_m3'):
                final_bbox_m3, final_scores_m3, final_category_m3 = self.postprocess_ssh(rois=rois_m3,
                                                                                         bbox_ppred=box_pred_m3,
                                                                                         scores=cls_prob_m3,
                                                                                         img_shape=img_shape,
                                                                                         iou_threshold=cfgs.M3_NMS_IOU_THRESHOLD)
            result_dict = {'final_bbox_m1': final_bbox_m1, 'final_scores_m1': final_scores_m1,
                           'final_category_m1': final_category_m1, 'final_bbox_m2': final_bbox_m2,
                           'final_scores_m2': final_scores_m2,  'final_category_m2': final_category_m2,
                           'final_bbox_m3': final_bbox_m3, 'final_scores_m3': final_scores_m3,
                           'final_category_m3': final_category_m3}
            losses_dict = {'bbox_loss_m1': bbox_loss_m1, 'cls_loss_m1': cls_loss_m1,
                           'bbox_loss_m2': bbox_loss_m2, 'cls_loss_m2': cls_loss_m2,
                           'bbox_loss_m3': bbox_loss_m3, 'cls_loss_m3': cls_loss_m3}
            return result_dict, losses_dict

    def enlarge_gradients_for_bias(self, gradients):
        final_gradients = []
        with tf.variable_scope("Gradient_Mult") as scope:
            for grad, var in gradients:
                scale = 1.0
                if cfgs.MUTILPY_BIAS_GRADIENT and './biases' in var.name:
                    scale = scale * cfgs.MUTILPY_BIAS_GRADIENT
                if not np.allclose(scale, 1.0):
                    grad = tf.multiply(grad, scale)
                final_gradients.append((grad, var))
        return final_gradients

if __name__ == '__main__':
    def get_imglabel(batch_size):
        img = np.ones([1,480,640,3],dtype=np.float32)
        label_gt = np.array([[10,10,100,100,0],[20,20,200,200,1],[50,50,300,300,1]],dtype=np.int32)
        return (img,label_gt)
    img_batch = tf.ones([1,480,640,3],dtype=tf.float32)
    #elem_list = tf.concat([elem,elem],0)
    #img_batch = tf.map_fn(lambda x:x+1,elem_list)
    #img_batch = tf.placeholder(tf.float32,shape=(1,480,640,3))
    gtboxes_and_label = tf.constant([[10,10,100,100,0],[20,20,200,200,1],[50,50,300,300,1]],dtype=tf.int32)
    #img_batch,gtboxes_and_label = tf.py_func(get_imglabel,[1],[tf.float32,tf.int32])
    faster_rcnn = DetectionNetwork(base_network_name=cfgs.NET_NAME,is_training=True)
    result_dict, losses_dict = faster_rcnn.build_ssh_network(input_img_batch=img_batch,gtboxes_batch=gtboxes_and_label)
    sess = tf.Session()
    summary_op = tf.summary.merge_all()
    wr = tf.summary.FileWriter('/home/lxy/Develop/Center_Loss/git_prj/SSH_prj/ssh-tensorflow/logs',sess.graph)