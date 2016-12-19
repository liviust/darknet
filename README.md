What does this fork do ?
========================

Darknet is a pretty simple implementation, so fun to poke at.
Here is some of my own notes on tooling

## Extracting Weights

tl;dr: check the weight_extraction dir

Reproduce the weight writing mechanism found in this forked copy of darknet :

https://github.com/gliese581gg/YOLO_tensorflow/tree/master/YOLO_weight_extractor

Usage :

    1. compile darknet

    2. run : darknet yolo write_weights cfg/tiny-yolo.cfg tiny-yolo.weights

It can also denormalize the weights if you don't want your forward
model to bother

Usage :

    run: darknet denormalize cfg/tiny-yolo.cfg tiny-yolo.weights tiny-yolo-denorm.weights

Note: I've modified the code so the denormalied weights still write
scales=1, rolling_mean=0, rolling_variance=1 in the saved weights, so it can
still be loaded without removing batch_normalization or loadscales in the cfg
for that net.


## Input image format / processing

IplImage (opencv) to Darknet image format

```
image ipl_to_image(IplImage* src)
{
    unsigned char *data = (unsigned char *)src->imageData;
    int h = src->height;
    int w = src->width;
    int c = src->nChannels;
    int step = src->widthStep;
    image out = make_image(w, h, c);
    int i, j, k, count=0;;

    for(k= 0; k < c; ++k){
        for(i = 0; i < h; ++i){
            for(j = 0; j < w; ++j){
                out.data[count++] = data[i*step + j*c + k]/255.;
            }
        }
    }
    return out;
}
```

image is organized
* TOCHECK: bgr or rgb same as Opencv
* shape: [channel, height, width]
* scaled 0 to 1



Original Readme
===============

![Darknet Logo](http://pjreddie.com/media/files/darknet-black-small.png)

#Darknet#
Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.

For more information see the [Darknet project website](http://pjreddie.com/darknet).

For questions or issues please use the [Google Group](https://groups.google.com/forum/#!forum/darknet).
