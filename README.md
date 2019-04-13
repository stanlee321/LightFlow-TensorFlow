# LightFlow Tensorflow implementation
Implementation of Optical Flow predictions using deep learning **LightFlow** architecture proposed in **[Towards High Performance Video Object Detection for Mobiles](https://arxiv.org/pdf/1804.05830.pdf)**

## Network Variables


``` bash


```
The main training procedure is based on **https://github.com/sampepose/flownet2-tf** , and the network was builded using Keras.

## Tensorbaord logs
![lightflow tensorboard Loss](/outputs/tensorboard.png?raw=true)
![lightflow tensorboard Val image](/outputs/output_val.png?raw=true)

### Actual input/output
![inputs](/data/samples/0img0.ppm?raw=true)
![inputs](/data/samples/0img1.ppm?raw=true)

![lightflow Actual output](/outputs/output.png?raw=true)


# TODOs
* Fix error with the BatchNorm layer in Keras, model does not converge.
* Migrate Keras model definition to TensorFlow
* Work in progress
# Reference article, thanks

* **https://github.com/sampepose/flownet2-tf**
* **https://arxiv.org/pdf/1804.05830.pdf**
* **https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py**