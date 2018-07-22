"""toolbox for plotting bbox on images
"""


def plot_image_bbox(self, image, boxes):
    """plot an image with its according boxes

    Args:
        images: an numpy array [r, g, b] format
        boxes: a list contains box
    """
    #image_name = os.path.basename(image_path)
    #image = cv2.imread(image_path)

    #label_name = os.path.splitext(image_name)[0] + '.txt'
    #labels = open(os.path.join(self.labels_path, label_name), 'r')
    #boxes = []
    #for line in labels.readlines():
    #    box = [float(x) for x in line.strip().split()]
    #    boxes.append(box)

    for box in boxes:
        shape = image.shape
        hight = shape[0]
        width = shape[1]

        cls_index, x, y, w, h = box

        x *= width
        w *= width
        y *= hight
        h *= hight

        #draw bbox
        top_left = (int(x - w / 2), int(y - h / 2))
        bottom_right = (int(x + w / 2), int(y + h / 2))
        cv2.rectangle(image, top_left, bottom_right, (255, 220, 33))

        #draw Text background rectangle
        text_size, baseline = cv2.getTextSize(classes[int(cls_index)], cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(image, (top_left[0], top_left[1] - text_size[1]),
                             (top_left[0] + text_size[0], top_left[1]),
                             (255, 220, 33),
                             -1)
        
        #draw text
        cv2.putText(image, classes[int(cls_index)], 
                           top_left,
                           cv2.FONT_HERSHEY_DUPLEX,
                           0.4,
                           (255, 255, 255),
                           1,
                           8,
                           )
    cv2.imshow('test', image)
    waitKey(0)