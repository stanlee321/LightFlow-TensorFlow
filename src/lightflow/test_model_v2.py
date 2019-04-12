from lightflow import LightFlow, Encoder, Decoder
import tensorflow as tf
from tensorflow.keras import Input

def main():
    # add model to graph


    concat_inputs = Input((384, 512, 6))

    encoder = Encoder()
    conv6b, conv5b, conv4b, conv3, conv2 = encoder(concat_inputs)

    decoder = Decoder()
    _ = decoder(conv2,conv3, conv4b, conv5b, conv6b )


    encoder.summary()
    decoder.summary()

    model = LightFlow()
    model(concat_inputs)

    model.summary()


if __name__ == '__main__':
    main()