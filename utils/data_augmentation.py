""" this module is the helper functon for data
augmentation

Author: baiyu
"""

import random

import cv2
import numpy as np

import utils.plot_tools as plot_tools
from conf import settings

def resize(image, boxes, image_shape):
    """resize image and boxes to certain
    shape

    Args:
        image: a numpy array(BGR)
        boxes: bounding boxes
        image_shape: two element tuple(width, height)
    
    Returns:
        resized image and boxes
    """

    origin_shape = image.shape
    x_factor = image_shape[1] / float(origin_shape[1])
    y_factor = image_shape[0] / float(origin_shape[0])

    #resize_image
    if (image.shape[1], image.shape[0]) != image_shape:
        image = cv2.resize(image, image_shape)

    #resize_box
    result = []
    for box in boxes:
        cls_id, x, y, w, h = plot_tools.unnormalize_box_params(box, origin_shape)

        x *= x_factor
        w *= x_factor
        y *= y_factor 
        h *= y_factor

        #clamping the box board, make sure box inside the image, 
        #not on the board
        tl_x = x - w / 2
        tl_y = y - h / 2
        br_x = x + w / 2
        br_y = y + h / 2

        tl_x = min(max(0, tl_x), settings.IMG_SIZE - 1)
        tl_y = min(max(0, tl_y), settings.IMG_SIZE - 1)
        br_x = max(min(settings.IMG_SIZE - 1, br_x), 0)
        br_y = max(min(settings.IMG_SIZE - 1, br_y), 0)

        w = br_x - tl_x
        h = br_y - tl_y
        x = (br_x + tl_x) / 2
        y = (br_y + tl_y) / 2

        Box = type(box)
        box = Box(cls_id, x, y, w, h)
        result.append(plot_tools.normalize_box_params(box, image.shape))

    return image, result 

def random_crop(image, boxes):
    """randomly crop image, resize image's
    shortest side to settings.IMG_SIZE * (1 + scale_factor)
    while remain the aspect ratio, then crop a 
    settings.IMG_SIZE * settings.IMG_SIZE image

    Args:
        image: a image numpy array(BGR)
        boxes: boxes corresponding to image

    Returns:
        (image, boxes): possible flipped image
        and boxes
    """

    if random.random() < settings.AUG_PROB:
        origin_shape = image.shape
        min_side = min(image.shape[:2])

        #resize the image
        resized_side = int(settings.IMG_SIZE * (1 + settings.CROP_JITTER))
        scale_ratio = resized_side / float(min_side)
        image = cv2.resize(image, (0, 0), fx=scale_ratio, fy=scale_ratio)
        
        for index, box in enumerate(boxes):
            cls_id, x, y, w, h = plot_tools.unnormalize_box_params(box, origin_shape)
    
            x *= scale_ratio 
            y *= scale_ratio
            w *= scale_ratio
            h *= scale_ratio

            Box = type(box)
            box = Box(cls_id, x, y, w, h)
            boxes[index] = plot_tools.normalize_box_params(box, image.shape)
    
        #crop the image
        mask = [[0, settings.IMG_SIZE], [0, settings.IMG_SIZE]]
        random_shift_x = random.randint(0, image.shape[1] - settings.IMG_SIZE)
        random_shift_y = random.randint(0, image.shape[0] - settings.IMG_SIZE)
        mask[0][0] = random_shift_x
        mask[0][1] = random_shift_x + settings.IMG_SIZE
        mask[1][0] = random_shift_y
        mask[1][1] = random_shift_y + settings.IMG_SIZE

        before_cropped = image.shape
        image = image[mask[1][0] : mask[1][1], mask[0][0] : mask[0][1], :]

        #crop boxes
        result = []
        for box in boxes:
            cls_id, x, y, w, h = plot_tools.unnormalize_box_params(box, before_cropped)

            #get old top_left, bottom_right coordinates
            old_tl_x = x - int(w / 2)
            old_tl_y = y - int(h / 2)
            old_br_x = x + int(w / 2)
            old_br_y = y + int(h / 2)

            #clamp the old box coordinates
            new_tl_x = min(max(old_tl_x, mask[0][0]), mask[0][1])
            new_tl_y = min(max(old_tl_y, mask[1][0]), mask[1][1])
            new_br_x = max(min(old_br_x, mask[0][1]), mask[0][0])
            new_br_y = max(min(old_br_y, mask[1][1]), mask[1][0])


            #get new w, h
            if new_br_x - new_tl_x <= 0:
                continue
            w = new_br_x - new_tl_x
            if new_br_y - new_tl_y <= 0:
                continue
            h = new_br_y - new_tl_y

            #get new x, y
            x = (new_br_x + new_tl_x) / 2 - mask[0][0] 
            y = (new_br_y + new_tl_y) / 2 - mask[1][0]

            Box = type(box)
            box = Box(cls_id, x, y, w, h)
            result.append(plot_tools.normalize_box_params(box, image.shape))

        boxes = result
    return image, boxes

def random_affine(image, boxes):
    """randomly apply affine transformation
    to an image

    Args:
        image: an image numpy array(BGR)
        boxes: boxes corresponding to image

    Returns:
        (image, boxes): possible flipped image
        and boxes
    """
    if random.random() < settings.AUG_PROB:
        height, width, _ = image.shape

        shift_x = int(width * random.uniform(0, settings.AFFINE_SHIFT_FACTOR))
        shift_y = int(height * random.uniform(0, settings.AFFINE_SHIFT_FACTOR))
        scale_x = float(random.uniform(settings.AFFINE_SCALE_FACTOR, 1))
        scale_y = float(random.uniform(settings.AFFINE_SCALE_FACTOR, 1))

        #affine translation matrix
        trans = np.array([[scale_x, 0, shift_x],
                          [0, scale_y, shift_y]], dtype=np.float32)

        image = cv2.warpAffine(image, trans, (width, height))
    
        #change boxes
        result = []
        for index, box in enumerate(boxes):
            cls_id, x, y, w, h = plot_tools.unnormalize_box_params(box, image.shape)
            x *= scale_x
            w *= scale_x
            y *= scale_y
            h *= scale_y

            x += shift_x
            y += shift_y

            Box = type(box)
            box = Box(cls_id, x, y, w, h)

            # if bounding box is still in the image
            # shift might shitf the bounding box
            # outside of the image
            if (width - (x - w / 2)) > 0 and (height - (y - h / 2)) > 0:
                result.append(plot_tools.normalize_box_params(box, image.shape))
        
        boxes = result
    return image, boxes

def random_horizontal_flip(image, boxes):
    """randomly flip an image left to right

    Args:
        image: a numpy array of a BGR image
        boxes: boxes corresponding to image

    Returns:
        (image, boxes): possible flipped image
        and boxes
    """

    if random.random() < settings.AUG_PROB:

        #flip image right to left
        image = cv2.flip(image, 1)

        #flip boxes
        image_shape = image.shape
        for index, box in enumerate(boxes):
            cls_num, x, y, w, h = plot_tools.unnormalize_box_params(box, image_shape)
            x = image_shape[1] - x
            Box = type(box)
            box = Box(cls_num, x, y, w, h)
            boxes[index] = plot_tools.normalize_box_params(box, image_shape)

    return image, boxes

def random_bright(image):
    """randomly brightten an image

    Args:
        image: an image numpy array(BGR)
    """

    if random.random() < settings.AUG_PROB:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        factor = random.uniform(1. - settings.BRIGHTNESS, 1. + settings.BRIGHTNESS)
        v = v * factor
        v = np.clip(v, 0, 255).astype(hsv.dtype)
        hsv = cv2.merge((h, s, v))
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return image

def random_hue(image):
    """randomly change the hue of an image

    Args:
        image: an image numpy array(BGR)
    """

    if random.random() < settings.AUG_PROB:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        factor = random.uniform(1. - settings.HUE_FACTOR, 1. + settings.HUE_FACTOR)
        h = h * factor
        h = np.clip(h, 0, 255).astype(hsv.dtype)
        hsv = cv2.merge((h, s, v))
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return image

def random_saturation(image):
    """randomly change the saturation of an image

    Args:
        image: an image numpy array(BGR)
    """

    if random.random() < settings.AUG_PROB:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v= cv2.split(hsv)
        factor = random.uniform(1. - settings.STATURATION_FACTOR, 
                                1. + settings.STATURATION_FACTOR)
        s = s * factor
        s = np.clip(s, 0, 255).astype(hsv.dtype)
        hsv = cv2.merge((h, s, v))
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return image

def random_gaussian_blur(image):
    """ randomly blurs an image using a Gaussian filter.

    Args:
        image: an image numpy array(BGR)
    """

    if random.random() < settings.AUG_PROB:
        image = cv2.GaussianBlur(image, (5, 5), 0)
    
    return image