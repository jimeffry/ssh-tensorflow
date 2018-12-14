#!/bin/bash
#python convert_data_to_tfrecord.py --VOC-dir /home/lxy/Downloads/DataSet/VOC_dataset/VOCdevkit/VOC2012  --xml-dir Annotations --image-dir JPEGImages \
 #           --save-name train --dataset-name VOC2012  
#widerface
python convert_data_to_tfrecord.py    --image-dir /home/lxy/Downloads/DataSet/Face_dec_db/WiderFace/WIDER_train/images \
            --save-name train --dataset-name WiderFace --anno-file /home/lxy/Develop/Center_Loss/git_prj/SSH_prj/ssh-tensorflow/data/wider_face_train.txt