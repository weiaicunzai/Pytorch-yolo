# Under Construction........

# pytorch implementation of yolov1

*here is the [paper](https://arxiv.org/abs/1506.02640v5)*

*write this code for learning purpose, learned from [this](https://github.com/xiongzihua/pytorch-YOLO-v1) repository*

*network architecture will be like [this](https://github.com/pjreddie/darknet/blob/master/cfg/yolov1.cfg)*

## Abstruct

Pytorch implementation of yolov1


## implementation details

- Image preprocessing: 


I don't use mean substracting and std deviation as the preprocessing tricks,
because not all the case we can know the mean and std of a dataset, for example,
a live camera video flow.Yolo source code and the paper both suggests that no
mean subtracting and std deviation applied:

The source code in [yolov2](https://github.com/pjreddie/darknet/blob/d3828827e70b293a3045a1eb80bfb4026095b87b/examples/yolo.c):
```c
    load_args args = {0};
    args.w = net->w;
    args.h = net->h;
    args.paths = paths;
    args.n = imgs;
    args.m = plist->size;
    args.classes = classes;
    args.jitter = jitter;
    args.num_boxes = side;
    args.d = &buffer;
    args.type = REGION_DATA;

    args.angle = net->angle;
    args.exposure = net->exposure;
    args.saturation = net->saturation;
    args.hue = net->hue;
```

In [yolov2](https://arxiv.org/abs/1612.08242v1) paper:
```
During training we use standard data augmentation tricks including random
crops, rotations, and hue, saturation, and exposure shifts.
```

- Image Channel

I'm using OpenCV as my image reading library, so the RGB order is BGR,
If you want to use my code to predict image, or videos, please change
your channel order to BGR

- Compatibility

I use Python3.5 to write the code, I've already tried my best to maintain
the compatibility as possible, e.g. the result int / int is float in Python3.5
I tried my best to avoid using / sign, but I Can't make any promises about
the compatibility. Please use Python3.5!