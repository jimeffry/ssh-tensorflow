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
#reversion:https://imgaug.readthedocs.io/en/latest/source/examples_bounding_boxes.html
####################################################
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
import cv2 

class Transform(object):
    def __init__(self,landmark_num=5,class_num=2,\
                img_auger_list=["Sequential", "SomeOf", "OneOf", "Sometimes"]):
        '''
        for image augmentation
        img_auger_list: "Sequential", "SomeOf", "OneOf", "Sometimes","Fliplr", "Flipud", "CropAndPad", \
                        "Affine", "PiecewiseAffine","Superpixels","GaussianBlur","AverageBlur","MedianBlur", \
                        "Sharpen", "Emboss","SimplexNoiseAlpha","AdditiveGaussianNoise","Dropout","Invert", \
                        "AddToHueAndSaturation", "Multiply","FrequencyNoiseAlpha","ContrastNormalization","Grayscale", \
                        "ElasticTransformation", "","PerspectiveTransform"
        '''
        self.landmark_num = landmark_num
        self.IMG_AUGMENTERS = img_auger_list
        self.DEBUG = 1
        self.Class_Num = class_num

    def hook(self,images, augmenter, parents, default):
            """Determines which augmenters to apply to masks."""
            return augmenter.__class__.__name__ in self.IMG_AUGMENTERS

    def aug_seq(self):
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        seq = iaa.Sequential(
            [
                # apply the following augmenters to most images
                iaa.Fliplr(0.5), # horizontally flip 50% of all images
                iaa.Flipud(0.2), # vertically flip 20% of all images
                # crop images by -5% to 10% of their height/width
                iaa.CropAndPad(
                    percent=(-0.05, 0.1),
                    pad_mode=ia.ALL,
                    pad_cval=(0, 255)
                ),
                iaa.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
                    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                    rotate=(-45, 45), # rotate by -45 to +45 degrees
                    shear=(-16, 16), # shear by -16 to +16 degrees
                    order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                    cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                    mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                ),
                # execute 0 to 5 of the following (less important) augmenters per image
                # don't execute all of them, as that would often be way too strong
                iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200)), # convert images into their superpixel representation
                iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                # search either for all edges or for directed edges,
                # blend the result with the original image using a blobby mask
                iaa.SimplexNoiseAlpha(iaa.OneOf([
                            iaa.EdgeDetect(alpha=(0.5, 1.0)),
                            iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                ])),
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                iaa.OneOf([
                    iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                    iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                ]),
                iaa.Invert(0.05, per_channel=True), # invert color channels
                iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
                # either change the brightness of the whole image (sometimes
                # per channel) or change the brightness of subareas
                iaa.OneOf([
                    iaa.Multiply((0.5, 1.5), per_channel=0.5),
                    iaa.FrequencyNoiseAlpha(
                        exponent=(-4, 0),
                        first=iaa.Multiply((0.5, 1.5), per_channel=True),
                        second=iaa.ContrastNormalization((0.5, 2.0))
                    )
                ]),
                iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                iaa.Grayscale(alpha=(0.0, 1.0)),
                sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
                sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1))),
            ],
            random_order=True
        )
        return seq


    def aug_img_boxes(self,images,bboxes):
        '''
        images:[img1,img2]
        bboxes:[[[x1,y1,x2,y2],...],[[x1,y1,x2,y2],...]]]
        '''
        if not isinstance(images,list):
            images = [images]
        assert len(images) == len(bboxes),"images list should equal to keypoints list"
        bboxes_on_images = []
        keep_indx = []
        idx = 0
        for image,cur_box in zip(images,bboxes):
            height, width = image.shape[0:2]
            boxes_one_img = []
            for i in range(len(cur_box)):
                x1 = int(cur_box[i][0])
                y1 = int(cur_box[i][1])
                x2 = int(cur_box[i][2])
                y2 = int(cur_box[i][3])
                boxes_one_img.append(ia.BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2))
            bboxes_on_images.append(ia.BoundingBoxesOnImage(boxes_one_img, shape=image.shape))
        #select actiate imgaugmenters
        seq = self.aug_seq()
        hook_activate = ia.HooksImages(activator=self.hook)
        seq_det = seq.to_deterministic()
        # Augment BBs and images.
        # As we only have one image and list of BBs, we use
        # [image] and [bbs] to turn both into lists (batches) for the
        # functions and then [0] to reverse that. In a real experiment, your
        # variables would likely already be lists.
        images_aug = seq_det.augment_images(images,hooks=hook_activate)
        bbs_aug = seq_det.augment_bounding_boxes(bboxes_on_images,hooks=hook_activate)
        # print coordinates before/after augmentation (see below)
        # use .x1_int, .y_int, ... to get integer coordinates
        img_out = []
        bbs_out = []
        for img_idx, (image_before, image_after, bbs_before, bbs_after) in \
                    enumerate(zip(images, images_aug, bboxes_on_images, bbs_aug)):
            #after augmentation and removing those fully outside the image and
            # cutting those partially inside the image so that they are fully inside.
            bbx_after = bbs_after.remove_out_of_image().cut_out_of_image()
            if self.DEBUG:
                image_before = bbs_before.draw_on_image(image_before)
                image_after = bbx_after.draw_on_image(image_after)
                ia.show_grid([image_before, image_after],rows=1,cols=2) # before and after
            img_out.append(image_after)
            boxes_one_img = []
            for kp_idx, bbs in enumerate(bbx_after.bounding_boxes):
                bb_old = bboxes_on_images[img_idx].bounding_boxes[kp_idx]
                x1_old, y1_old,x2_old,y2_old = bb_old.x1,bb_old.y1,bb_old.x2,bb_old.y2
                x1_new, y1_new,x2_new, y2_new = bbs.x1, bbs.y1,bbs.x2, bbs.y2
                boxes_one_img.append([x1_new, y1_new,x2_new, y2_new])
                if self.DEBUG:
                    print("BB %d: (%.4f, %.4f, %.4f, %.4f) -> (%.4f, %.4f, %.4f, %.4f)" % (
                        kp_idx,
                        x1_old, y1_old,x2_old,y2_old,
                        x1_new, y1_new,x2_new, y2_new)
                    )
            bbs_out.append(boxes_one_img)
        return img_out,bbs_out
        
        
    def aug_img_keypoints(self,images,keypoints):
        '''
        images:[img1,img2]
        keypoints:[[[x1,y1],[x2,y2],...],[[x1,y1],[x2,y2],...]]]
        '''
        if not isinstance(images,list):
            images = [images]
        assert len(images) == len(keypoints),"images list should equal to keypoints list"
        keypoints_on_images = []
        images_aug_list = []
        keep_indx = []
        idx = 0
        for image,cur_points in zip(images,keypoints):
            height, width = image.shape[0:2]
            keypoints = []
            break_fg = 1
            for i in range(self.landmark_num):
                x = int(cur_points[i][0])
                y = int(cur_points[i][1])
                if x >width-1 or y >height-1 :
                    break_fg = 0
                keypoints.append(ia.Keypoint(x=x, y=y))
            if not break_fg :
                continue
            images_aug_list.append(image)
            keypoints_on_images.append(ia.KeypointsOnImage(keypoints, shape=image.shape))
            keep_indx.append(idx)
            idx+=1
        seq = self.aug_seq()
        hook_activate = ia.HooksImages(activator=self.hook)
        seq_det = seq.to_deterministic() # call this for each batch again, NOT only once at the start
        # augment keypoints and images
        images_aug = seq_det.augment_images(images_aug_list,hooks=hook_activate)
        keypoints_aug = seq_det.augment_keypoints(keypoints_on_images,hooks=hook_activate)
        img_out = []
        keypoint_out = []
        for img_idx, (image_before, image_after, keypoints_before, keypoints_after) in \
                    enumerate(zip(images_aug_list, images_aug, keypoints_on_images, keypoints_aug)):
            if self.DEBUG:
                image_before = keypoints_before.draw_on_image(image_before)
                image_after = keypoints_after.draw_on_image(image_after)
                ia.show_grid([image_before, image_after],rows=1,cols=2) # before and after
            img_out.append(image_after)
            key_img = []
            for kp_idx, keypoint in enumerate(keypoints_after.keypoints):
                keypoint_old = keypoints_on_images[img_idx].keypoints[kp_idx]
                x_old, y_old = keypoint_old.x, keypoint_old.y
                x_new, y_new = keypoint.x, keypoint.y
                key_img.append([x_new,y_new])
                if self.DEBUG:
                    print("[Keypoints for image #%d] before aug: x=%d y=%d | \
                            after aug: x=%d y=%d" % (img_idx, x_old, y_old, x_new, y_new))
            keypoint_out.append(key_img)
        return img_out,keypoint_out,keep_indx

    def aug_img_masks(self,images,masks):
        '''
        images:[img1,img2]
        masks:[m1,m2]
        '''
        if not isinstance(images,list):
            images = [images]
        if not isinstance(masks,list):
            masks = [masks]
        masks_on_images = []
        for cur_img,cur_mask in zip(images,masks):
            segmap = ia.SegmentationMapOnImage(cur_mask, shape=cur_img.shape, nb_classes=self.Class_Num)
            masks_on_images.append(segmap)
        seq = self.aug_seq()
        hook_activate = ia.HooksImages(activator=self.hook)
        seq_det = seq.to_deterministic() # call this for each batch again, NOT only once at the start
        # augment masks and images
        images_aug = seq_det.augment_images(images,hooks=hook_activate)
        segmaps_aug = seq_det.augment_segmentation_maps(masks_on_images,hooks=hook_activate)
        img_out = []
        masks_out = []
        cells = []
        for img_idx,(image_aug, segmap_aug) in enumerate(zip(images_aug, segmaps_aug)):
            img_out.append(image_aug)                                  # column 3
            masks_out.append(segmap_aug)
            if self.DEBUG:
                cells.append(images[img_idx])                            # column 1
                cells.append(masks_on_images[img_idx].draw_on_image(images[img_idx]))   # column 2
                cells.append(image_aug)                                  # column 3
                cells.append(segmap_aug.draw_on_image(image_aug))        # column 4
                cells.append(segmap_aug.draw(size=image_aug.shape[:2]))  # column 5
        # Convert cells to grid image and save
        if self.DEBUG:
            grid_image = ia.draw_grid(cells, cols=5)
            imageio.imwrite("example_segmaps.jpg", grid_image)
        return img_out,masks_out

    def aug_img(self,images):
        if not isinstance(images,list):
            images = [images]
        seq = self.aug_seq()
        hook_activate = ia.HooksImages(activator=self.hook)
        seq_det = seq.to_deterministic() # call this for each batch again, NOT only once at the start
        # augment images
        images_aug = seq_det.augment_images(images,hooks=hook_activate)
        return images_aug

if __name__=='__main__':
    auger_list=["Sequential", "Fliplr", "Flipud", "CropAndPad","Affine", "PiecewiseAffine"]
    trans = Transform(landmark_num=5,img_auger_list=auger_list)
    img = cv2.imread("/home/lxy/Develop/Center_Loss/git_prj/SSH_prj/ssh-tensorflow/test.jpg")
    box1=[10,10,100,100]
    box2=[200,200,350,350]
    boxes=[[box1,box2]]
    points = [[[200,200],[300,200],[250,250],[200,300],[300,300]]]
    mk = np.zeros(img.shape)
    mk[100:150,200:250,:] = 10
    #trans.aug_img_boxes(img,boxes)
    #trans.aug_img_keypoints(img,points)
    trans.aug_img_masks(img,mk)