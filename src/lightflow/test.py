import argparse
import os
from ..net import Mode
from .lightflow_tf import LightFlow
FLAGS = None

"""
run with
python -m  src.lightflow.test \
--input_a data/samples/0img0.ppm  \
--input_b data/samples/0img1.ppm --out ./
"""
def main_test():
    # Create a new network
    net = LightFlow(mode=Mode.TEST)

    # Train on the data
    net.test(
        checkpoint= './checkpoints/lightflow/model.ckpt-75000', #"./checkpoints_old/lightflow/model.ckpt-9468"
        input_a_path=FLAGS.input_a,
        input_b_path=FLAGS.input_b,
        out_path=FLAGS.out,
    )

def main_build():
    # Create a new network
    net = LightFlow(mode=Mode.TEST)

    # Train on the data
    net.test_ckpt(
        checkpoint='./checkpoints/lightflow/model.ckpt-75000',
        input_a_path=FLAGS.input_a,
        input_b_path=FLAGS.input_b,
        output_path='./checkpoints/model')
    
def main_cam():
    # Create a new network
    net = LightFlow(mode=Mode.TEST)
    # Train on the data
    net.test_cam(
        checkpoint='./checkpoints/model/lightflow_fixed_optimized.pb'
    )
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_a',
        type=str,
        required=True,
        help='Path to first image'
    )
    parser.add_argument(
        '--input_b',
        type=str,
        required=True,
        help='Path to second image'
    )
    parser.add_argument(
        '--out',
        type=str,
        required=True,
        help='Path to output flow result'
    )
    FLAGS = parser.parse_args()

    # Verify arguments are valid
    if not os.path.exists(FLAGS.input_a):
        raise ValueError('image_a path must exist')
    if not os.path.exists(FLAGS.input_b):
        raise ValueError('image_b path must exist')
    if not os.path.isdir(FLAGS.out):
        raise ValueError('out directory must exist')
    main_test()