from ..utils.dataloader import load_batch
from ..utls.utils import average_endpoint_error
from ..utils.dataset_configs import FLYING_CHAIRS_DATASET_CONFIG
from lightflow import LightFlow
from scheduler import CustomSchedule

# Create a new network
model = LightFlow()


# Create LR Schedules
learning_rate = CustomSchedule()


# Create Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, 
                                     epsilon=1e-9)

# Loss and Metrics


def loss_function(self, flow, predictions):

    # L2 loss between predict_flow, concat_input(img_a,img_b)
    predicted_flow = predictions

    size = [predicted_flow.shape[1], predicted_flow.shape[2]]

    downsampled_flow = downsample(flow, size)

    loss = average_endpoint_error(downsampled_flow, predicted_flow)

    tf.losses.add_loss(loss)

    # Return the 'total' loss: loss fns + regularization terms defined in the model
    return tf.losses.get_total_loss()



# Set metrics
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


checkpoint_path = "./checkpoints/train" 

ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print ('Latest checkpoint restored!!')

EPOCHS = 20


@tf.function
def train_step(input_a, input_b, flow):

    tf.summary.image("image_a", input_a, max_outputs=2)
    tf.summary.image("image_b", input_b, max_outputs=2)
    
    concat = tf.concat([input_a, input_b], axis=3)
    with tf.GradientTape() as tape:

        # Forwad propagation
        predictions, _ = model(concat)

        # Get the loss
        loss = loss_function(flow, predictions):


    gradients = tape.gradient(loss, model.trainable_variables)    
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(flow, predictions)
    
    # plot learning rate
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('Learning Rate', learning_rate)
    

    # Pred Flow img0
    pred_flow_0 = predictions['flow'][0, :, :, :]
    pred_flow_0 = tf.py_func(flow_to_image, [pred_flow_0], tf.uint8)
    # Pred Flow img1
    pred_flow_1 = predictions['flow'][1, :, :, :]
    pred_flow_1 = tf.py_func(flow_to_image, [pred_flow_1], tf.uint8)

    pred_flow_img = tf.stack([pred_flow_0, pred_flow_1], 0)
    tf.summary.image('pred_flow', pred_flow_img, max_outputs=2)


    # True flow img0
    true_flow_0 = flow[0, :, :, :]
    true_flow_0 = tf.py_func(flow_to_image, [true_flow_0], tf.uint8)
    # True flow img1
    true_flow_1 = flow[1, :, :, :]
    true_flow_1 = tf.py_func(flow_to_image, [true_flow_1], tf.uint8)

    true_flow_img = tf.stack([true_flow_0, true_flow_1], 0)
    tf.summary.image('true_flow', true_flow_img, max_outputs=2)


@tf.function
def test_step(image, label):
    predictions = model(image)
    t_loss = loss_object(label, predictions)

    test_loss(t_loss)
    test_accuracy(label, predictions)



"""
for epoch in range(EPOCHS):
    start = time.time()

    train_loss.reset_states()
    train_accuracy.reset_states()
    
    # Load a batch of data

    input_a, input_b, flow = load_batch(FLYING_CHAIRS_DATASET_CONFIG, 'train', epoch)

    train_step(inp, tar)


    if (epoch + 1) % 5 == 0:
        print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
            epoch + 1, train_loss.result(), train_accuracy.result()))

        ckpt_save_path = ckpt_manager.save()
        print('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                            ckpt_save_path))

    print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                        train_loss.result(),
                                                        train_accuracy.result()))

    print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))


def evaluate(inp_sentence):
    start_token = [tokenizer_pt.vocab_size]
    end_token = [tokenizer_pt.vocab_size + 1]

    # inp sentence is portuguese, hence adding the start and end token
    inp_sentence = start_token + tokenizer_pt.encode(inp_sentence) + end_token
    encoder_input = tf.expand_dims(inp_sentence, 0)

    # as the target is english, the first word to the transformer should be the
    # english start token.
    decoder_input = [tokenizer_en.vocab_size]
    output = tf.expand_dims(decoder_input, 0)

    for i in range(MAX_LENGTH):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
            encoder_input, output)

        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, attention_weights = transformer(encoder_input,
                                                    output,
                                                    False,
                                                    enc_padding_mask,
                                                    combined_mask,
                                                    dec_padding_mask)

        # select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # return the result if the predicted_id is equal to the end token
        if tf.equal(predicted_id, tokenizer_en.vocab_size+1):
            return tf.squeeze(output, axis=0), attention_weights

        # concatentate the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0), attention_weights
