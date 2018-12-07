# LightFlow Tensorflow/Keras implementation
Implementation of Optical Flow predictions throught deep learning using LightFlow architecture proposed in **https://arxiv.org/pdf/1804.05830.pdf**

The main training procedure is based on **https://github.com/sampepose/flownet2-tf** , and the network was builded using keras Keras.

## Tensorbaord logs
![lightflow tensorboard Loss](/outputs/tensorboard.png?raw=true)
![lightflow tensorboard Val image](/outputs/output_val.png?raw=true)

## Actual output
![lightflow Actual output](/outputs/output.png?raw=true)


# TODOs
* Fix error with the BatchNorm layer in Keras, model does not converge.

# Reference article, thanks

**https://github.com/sampepose/flownet2-tf**
**https://arxiv.org/pdf/1804.05830.pdf**