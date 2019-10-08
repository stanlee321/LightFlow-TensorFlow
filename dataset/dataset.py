import tensorflow as tf
import numpy as np
import copy

from  imageHandler import Image
# tf.enable_eager_execution()
slim = tf.contrib.slim

def _get_dataset(dataset_config, split_name):
    """
    dataset_config: A dataset_config defined in datasets.py
    split_name: 'train'/'validate'
    """
    with tf.name_scope('_get_dataset'):
        if split_name not in dataset_config['SIZES']:
            raise ValueError('split name %s not recognized' % split_name)

        IMAGE_HEIGHT, IMAGE_WIDTH = dataset_config['IMAGE_HEIGHT'], dataset_config['IMAGE_WIDTH']
        reader = tf.TFRecordReader

        keys_to_features = {
            'image_a':tf.io.FixedLenFeature((), tf.string),
            'image_b': tf.io.FixedLenFeature((), tf.string),
            'flow': tf.io.FixedLenFeature((), tf.string),
        }

        items_to_handlers = {
            'image_a': Image(
                image_key='image_a',
                dtype=tf.float64,
                shape=[IMAGE_HEIGHT, IMAGE_WIDTH, 3],
                channels=3),
            'image_b': Image(
                image_key='image_b',
                dtype=tf.float64,
                shape=[IMAGE_HEIGHT, IMAGE_WIDTH, 3],
                channels=3),
            'flow': Image(
                image_key='flow',
                dtype=tf.float32,
                shape=[IMAGE_HEIGHT, IMAGE_WIDTH, 2],
                channels=2),
        }
        decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

        return slim.dataset.Dataset(
            data_sources=dataset_config['PATHS'][split_name],
            reader=reader,
            decoder=decoder,
            num_samples=dataset_config['SIZES'][split_name],
            items_to_descriptions=dataset_config['ITEMS_TO_DESCRIPTIONS'])


def _generate_coeff(param, discount_coeff=tf.constant(1.0), default_value=tf.constant(0.0)):

    """
    Data Aumentation Layer

    """
    if not all(name in param for name in ['rand_type', 'exp', 'mean', 'spread', 'prob']):
        raise RuntimeError('Expected rand_type, exp, mean, spread, prob in `param`')

    rand_type = param['rand_type']
    exp = float(param['exp'])
    mean = tf.convert_to_tensor(param['mean'], dtype=tf.float32)
    spread = float(param['spread'])  # AKA standard deviation
    prob = float(param['prob'])

    # Multiply spread by our discount_coeff so it changes over time
    spread = spread * discount_coeff

    if rand_type == 'uniform':
        value = tf.cond(spread > 0.0,
                        lambda: tf.random_uniform([], mean - spread, mean + spread),
                        lambda: mean)
        if exp:
            value = tf.exp(value)
    elif rand_type == 'gaussian':
        value = tf.cond(spread > 0.0,
                        lambda: tf.random_normal([], mean, spread),
                        lambda: mean)
        if exp:
            value = tf.exp(value)
    elif rand_type == 'bernoulli':
        if prob > 0.0:
            value = tf.contrib.distributions.Bernoulli(probs=prob).sample([])
        else:
            value = 0.0
    elif rand_type == 'uniform_bernoulli':
        tmp1 = 0.0
        tmp2 = 0
        if prob > 0.0:
            tmp2 = tf.contrib.distributions.Bernoulli(probs=prob).sample([])
        else:
            tmp2 = 0

        if tmp2 == 0:
            if default_value is not None:
                return default_value
        else:
            tmp1 = tf.cond(spread > 0.0,
                           lambda: tf.random_uniform([], mean - spread, mean + spread),
                           lambda: mean)
        if exp:
            tmp1 = tf.exp(tmp1)
        value = tmp1
    elif rand_type == 'gaussian_bernoulli':
        tmp1 = 0.0
        tmp2 = 0
        if prob > 0.0:
            tmp2 = tf.contrib.distributions.Bernoulli(probs=prob).sample([])
        else:
            tmp2 = 0

        if tmp2 == 0:
            if default_value is not None:
                return default_value
        else:
            tmp1 = tf.cond(spread > 0.0,
                           lambda: tf.random_normal([], mean, spread),
                           lambda: mean)
        if exp:
            tmp1 = tf.exp(tmp1)
        value = tmp1
    else:
        raise ValueError('Unknown distribution type %s.' % rand_type)
    return value

def load_batch(dataset_config, split_name, global_step):
    num_threads = 32
    reader_kwargs = {'options':  tf.io.TFRecordOptions(
        tf.compat.v1.python_io.TFRecordCompressionType.ZLIB)}

    with tf.name_scope('load_batch'):
        dataset = _get_dataset(dataset_config, split_name)
        data_provider = slim.dataset_data_provider.DatasetDataProvider(
                                    dataset,
                                    num_readers=num_threads,
                                    common_queue_capacity=2048,
                                    common_queue_min=1024,
                                    reader_kwargs=reader_kwargs)
        print(data_provider)          
        image_a, image_b, flow = data_provider.get(['image_a', 'image_b', 'flow'])

        print(image_a.shape)
        image_a, image_b, flow = map(tf.to_float, [image_a, image_b, flow])
        
        if dataset_config['PREPROCESS']['scale']:
            image_a = image_a / 255.0
            image_b = image_b / 255.0
        image_as, image_bs, flows = map(lambda x: tf.expand_dims(x, 0), [image_a, image_b, flow])
        
        
        # Perform data augmentation on GPU
        with tf.device('/cpu:0'):

            noise_coeff_a = None
            noise_coeff_b = None

            # Generate and apply noise coeff for A if defined in A params
            if 'noise' in dataset_config['PREPROCESS']['image_a']:
                discount_coeff = tf.constant(1.0)
                if 'coeff_schedule_param' in dataset_config['PREPROCESS']['image_a']:
                    initial_coeff = dataset_config['PREPROCESS']['image_a']['coeff_schedule_param']['initial_coeff']
                    final_coeff = dataset_config['PREPROCESS']['image_a']['coeff_schedule_param']['final_coeff']
                    half_life = dataset_config['PREPROCESS']['image_a']['coeff_schedule_param']['half_life']
                    discount_coeff = initial_coeff + \
                        (final_coeff - initial_coeff) * \
                        (2.0 / (1.0 + np.exp(-1.0986 * global_step / half_life)) - 1.0)

                noise_coeff_a = _generate_coeff(
                    dataset_config['PREPROCESS']['image_a']['noise'], discount_coeff)
                noise_a = tf.random_normal(shape=tf.shape(image_as),
                                           mean=0.0, stddev=noise_coeff_a,
                                           dtype=tf.float32)
                image_as = tf.clip_by_value(image_as + noise_a, 0.0, 1.0)

            # Generate noise coeff for B if defined in B params
            if 'noise' in dataset_config['PREPROCESS']['image_b']:
                discount_coeff = tf.constant(1.0)
                if 'coeff_schedule_param' in dataset_config['PREPROCESS']['image_b']:
                    initial_coeff = dataset_config['PREPROCESS']['image_b']['coeff_schedule_param']['initial_coeff']
                    final_coeff = dataset_config['PREPROCESS']['image_b']['coeff_schedule_param']['final_coeff']
                    half_life = dataset_config['PREPROCESS']['image_b']['coeff_schedule_param']['half_life']
                    discount_coeff = initial_coeff + \
                        (final_coeff - initial_coeff) * \
                        (2.0 / (1.0 + np.exp(-1.0986 * global_step / half_life)) - 1.0)
                noise_coeff_b = _generate_coeff(
                    dataset_config['PREPROCESS']['image_b']['noise'], discount_coeff)

            # Combine coeff from a with coeff from b
            if noise_coeff_a is not None:
                if noise_coeff_b is not None:
                    noise_coeff_b = noise_coeff_a * noise_coeff_b
                else:
                    noise_coeff_b = noise_coeff_a

            # Add noise to B if needed
            if noise_coeff_b is not None:
                noise_b = tf.random_normal(shape=tf.shape(image_bs),
                                           mean=0.0, stddev=noise_coeff_b,
                                           dtype=tf.float32)
                image_bs = tf.clip_by_value(image_bs + noise_b, 0.0, 1.0)

            


            return tf.train.batch([image_as, image_bs, flows],
                                  enqueue_many=True,
                                  batch_size=dataset_config['BATCH_SIZE'],
                                  capacity=dataset_config['BATCH_SIZE'] * 4,
                                  num_threads=num_threads,
                                  allow_smaller_final_batch=False)