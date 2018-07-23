"""toolbox for plotting bbox on images
"""

import cv2

from conf import settings

def unnormalize_box_params(box, image_shape):
    """a helper function to calculate the box
    parameters

    Args:
        box: a namedtuple contains box parameters in
            the format of cls_num, x, y, w, h
        image_shape: corresponding image shape
    
    Returnes:
        Box: a namedtuple of (cls_name, x, y, w, h)
    """

    cls_name, x, y, w, h = box
    height, width = image_shape[:2]

    #unnormalize
    x *= width
    w *= width
    y *= height
    h *= height

    Box = type(box)
    return Box(cls_name, x, y, w, h)

def normalize_box_params(box, image_shape):
    """a helper function to normalize the
    box params

    Args:
        box: a namedtuple contains box parameters in
            the format of cls_num, x, y, w, h
        image_shape: corresponding image shape (h, w, c)
    
    Returnes:
        Box: a namedtuple of (cls_name, x, y, w, h)
    """
    cls_name, x, y, w, h = box
    height, width = image_shape[:2]

    #normalize
    x *= 1. / width
    w *= 1. / width
    y *= 1. / height
    h *= 1. / height

    Box = type(box)
    return Box(cls_name, x, y, w, h)

def plot_image_bbox(image, boxes):
    """plot an image with its according boxes
    a useful test tool for 

    Args:
        images: an numpy array [r, g, b] format
        boxes: a list contains box
    """

    for box in boxes:
        shape = image.shape
        cls_index, x, y, w, h = unnormalize_box_params(box, shape)

        #draw bbox
        top_left = (int(x - w / 2), int(y - h / 2))
        bottom_right = (int(x + w / 2), int(y + h / 2))
        cv2.rectangle(image, top_left, bottom_right, (255, 220, 33))

        #draw Text background rectangle
        text_size, baseline = cv2.getTextSize(settings.CLASSES[int(cls_index)], 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(image, (top_left[0], top_left[1] - text_size[1]),
                             (top_left[0] + text_size[0], top_left[1]),
                             (255, 220, 33),
                             -1)
        
        #draw text
        cv2.putText(image, settings.CLASSES[int(cls_index)], 
                           top_left,
                           cv2.FONT_HERSHEY_DUPLEX,
                           0.4,
                           (255, 255, 255),
                           1,
                           8,
                           )
    cv2.imshow('test', image)
    cv2.waitKey(0)