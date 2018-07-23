""" this module is the helper functon for data
augmentation

Author: baiyu
"""

import random

import cv2
import numpy as np

import utils.plot_tools as plot_tools
from conf import settings

def random_crop(image, boxes):
    """randomly crop image, resize image's
    shortest side to 448 * (1 + scale_factor)
    while remain the aspect ratio, then crop a 
    448 * 448 image

    Args:
        image: a image numpy array(BGR)
        boxes: boxes corresponding to image

    Returns:
        (image, boxes): possible flipped image
        and boxes
    """

    #height, width = image.shape[:2]
    #aspect_ratio = height / float(width)
    if random.random() < settings.AUG_PROB:

        min_side = min(image.shape[:2])
        #min_index = image.shape[:2].index(min_side)
    
        #resize the image
        resized_side = int(448 * (1 + settings.CROP_JITTER))
        scale_ratio = resized_side / float(min_side)
        image = cv2.resize(image, (0, 0), fx=scale_ratio, fy=scale_ratio)
        
        for index, box in enumerate(boxes):
            cls_id, x, y, w, h = plot_tools.unnormalize_box_params(box, image.shape)
    
            x = x * scale_ratio * (1 + scale_ratio)
            y = y * scale_ratio * (1 + scale_ratio)
            w = w * scale_ratio * (1 + scale_ratio)
            h = h * scale_ratio * (1 + scale_ratio)
    
            Box = type(box)
            box = Box(cls_id, x, y, w, h)
            boxes[index] = plot_tools.normalize_box_params(box, image.shape)
    
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