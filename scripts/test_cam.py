import cv2
import os
import sys
from .flowprocessor import ImageProcessor
from .flowlib import flow_to_image, write_flow
import tensorflow as tf
import time

def test_cam(checkpoint):
    height =  384
    width = 512
    cap = cv2.VideoCapture(0)
    # if video file successfully open then read an initial frame from video
    for i in range(5):
        ret, input_a = cap.read()
    detect_flow = ImageProcessor(path_to_model=checkpoint)
    detect_flow.setup()
    while True:

        # if video file successfully open then read frame from video

        ret, input_b = cap.read()

        input_a = cv2.resize(input_a, (width, height))
        input_b = cv2.resize(input_b, (width, height))

        # Calculate the flow
        t1 = time.time()
        predictions = detect_flow.detect(input_a, input_b)
        print("elapsedtime =", time.time() - t1)
        flow_img = flow_to_image(predictions)
        input_a = input_b

        # display image with optic flow overlay

        cv2.imshow('flow', cv2.resize(flow_img, (640,480)))

        key = cv2.waitKey(40) & 0xFF; # wait 40ms (i.e. 1000ms / 25 fps = 40 ms)

        if (key == ord('q')):
            break
    # close all windows
    cv2.destroyAllWindows()


if __name__ == '__main__':
    cwd = os.getcwd()
    checkpoint=os.path.join(cwd, 'checkpoints/model_75k/lightflow_fixed_optimized.pb')
    print(checkpoint)
    test_cam(checkpoint = checkpoint)