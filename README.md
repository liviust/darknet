What does this fork do ?
=======================

Reproduce the weight writing mechanism found in this forked copy of darknet :

https://github.com/gliese581gg/YOLO_tensorflow/tree/master/YOLO_weight_extractor

Usage :

    1. compile darknet

    2. run : darknet yolo write_weights cfg/tiny-yolo.cfg tiny-yolo.weights

It can also denormalize the weights if you don't want your forward
model to bother

Usage :

    run: darknet cfg/tiny-yolo.cfg tiny-yolo.weights tiny-yolo-denorm.weights

Note: I've modified the code so the denormalied weights still write
scales=1, rolling_mean=0, rolling_variance=1 in the saved weights, so it can
still be loaded without removing batch_normalization or loadscales in the cfg
for that net.

Original Readme
===============

![Darknet Logo](http://pjreddie.com/media/files/darknet-black-small.png)

#Darknet#
Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.

For more information see the [Darknet project website](http://pjreddie.com/darknet).

For questions or issues please use the [Google Group](https://groups.google.com/forum/#!forum/darknet).
