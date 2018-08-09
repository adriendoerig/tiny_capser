from time import time
from tiny_capser_capsule_functions import *

def model_fn(features, labels, mode, params):

    simultaneous_shapes = 2

    # get inputs from .trecords file.
    X = features['X']                                                                   # input image
    x_image = tf.reshape(X, [params['model_batch_size'], im_size[0], im_size[1], 1])    # for tensorboard
    tf.summary.image('input', x_image, 6)                                               # for tensorboard
    reconstruction_targets = features['reconstruction_targets']                         # contains two images: one for the vernier in this stim, and one for the other shape
    vernier_offsets = features['vernier_offsets']                                       # vernier offsets
    y = tf.cast(features['y'], tf.int64)                                                # labels
    mask_with_labels = tf.placeholder_with_default(features['mask_with_labels'], shape=(), name="mask_with_labels")     # tell the program whether to use the true or the predicted labels (the placeholder is needed to have a bool in tf) -> cf. aurélien géron's video.
    is_training = True                                                                                                  # this is a fithy trick that has to do with batch norm. Don't worry about it for now.
    print_shapes = False                                                                                                # to print the size of each layer during graph construction


    ####################################################################################################################
    # Early conv layers and first capsules
    ####################################################################################################################


    with tf.name_scope('0_early_conv_layers'):
        # sizes, etc.
        conv1_width = int((im_size[0] - conv1_params["kernel_size"]) / conv1_params["strides"] + 1)
        conv1_height = int((im_size[1] - conv1_params["kernel_size"]) / conv1_params["strides"] + 1)

        conv2_width = int((conv1_width - conv2_params["kernel_size"]) / conv2_params["strides"] + 1)
        conv2_height = int((conv1_height - conv2_params["kernel_size"]) / conv2_params["strides"] + 1)

        caps1_n_caps = int((caps1_n_maps *
                            int((conv2_width - conv_caps_params["kernel_size"]) / conv_caps_params[
                                "strides"] + 1) *
                            int((conv2_height - conv_caps_params["kernel_size"]) / conv_caps_params[
                                "strides"] + 1)))


        # create early conv layers
        conv1 = tf.layers.conv2d(X, name="conv1", **conv1_params)
        tf.summary.histogram('1st_conv_layer', conv1)
        conv2 = tf.layers.conv2d(conv1, name="conv2", **conv2_params)
        tf.summary.histogram('2nd_conv_layer', conv2)


    with tf.name_scope('1st_caps'):

        # create the first layer of capsules (cf. aurélien géron's video)
        caps1_output, caps1_output_with_maps = primary_caps_layer(conv2, caps1_n_maps, caps1_n_caps,
                                                                  caps1_n_dims,
                                                                  conv_caps_params["kernel_size"],
                                                                  conv_caps_params["strides"],
                                                                  conv_padding=conv_caps_params['padding'],
                                                                  conv_activation=conv_caps_params[
                                                                      'activation'],
                                                                  print_shapes=print_shapes)

        # display a histogram of primary capsule norms
        caps1_output_norms = safe_norm(caps1_output, axis=-1, keep_dims=False, name="primary_capsule_norms")
        tf.summary.histogram('Primary capsule norms', caps1_output_norms)



    ####################################################################################################################
    # From caps1 to caps2
    ####################################################################################################################


    with tf.name_scope('2nd_caps'):

        # create the first layer of capsules (cf. aurélien géron's video)
        caps2_output = primary_to_fc_caps_layer(X, caps1_output, caps1_n_caps, caps1_n_dims, caps2_n_caps, caps2_n_dims, rba_rounds=rba_rounds, print_shapes=print_shapes)

        # get norms of all capsules for the first simulus in the batch to vizualize them
        caps2_output_norm = tf.squeeze(safe_norm(caps2_output[0, :, :, :], axis=-2, keep_dims=False, name="caps2_output_norm"))
        tf.summary.histogram('Output capsule norms', caps2_output_norm)

        # Estimated class probabilities
        y_pred = caps_prediction(caps2_output, n_labels=len(y.shape), print_shapes=print_shapes)  # get index of max probability

        # Compute the margin loss
        margin_loss = compute_margin_loss(y, caps2_output, caps2_n_caps, m_plus, m_minus, lambda_, print_shapes=print_shapes)


    ####################################################################################################################
    # Decoders: vernier offset loss & reconstruction loss
    ####################################################################################################################


    with tf.name_scope('decoders'):

        # vernier offset loss
        with tf.name_scope('vernier_offset_loss'):
            training_vernier_decoder_input = caps2_output[:, 0, 0, :, 0]  # decode from vernier capsule
            training_vernier_loss, vernier_accuracy, vernier_logits = compute_vernier_offset_loss(
                training_vernier_decoder_input, vernier_offsets, print_shapes)


        with tf.name_scope('reconstruction_loss'):

            decoder_outputs = []

            # First, create decoded images. We will decode one reconstruction for each present shape (using batch norm).
            for this_shape in range(2):

                with tf.variable_scope('shape_' + str(this_shape)):
                    # first, set all uninteresting capsules to 0 (cf, aurélien géron's video)
                    this_masked_output = create_masked_decoder_input(y[:, this_shape], y_pred[:, this_shape],
                                                                     caps2_output, caps2_n_caps, caps2_n_dims,
                                                                     mask_with_labels, print_shapes=print_shapes)

                    # then, use the masked caps2_output as input to the decoder
                    decoder_outputs.append(decoder_with_mask_batch_norm(this_masked_output, im_size[0] * im_size[1],
                                                                        output_caps_decoder_n_hidden1,
                                                                        output_caps_decoder_n_hidden2, phase=is_training,
                                                                        name='output_decoder'))

            decoder_outputs = tf.stack(decoder_outputs)                     # put decoded images for each shape in the same tensor
            decoder_outputs = tf.transpose(decoder_outputs, [1, 2, 0])      # technicality. don't worry.
            if print_shapes:
                print('shape of decoder_outputs: ' + str(decoder_outputs))

            # Compute reconstruction loss. We will compute one loss for each present shape
            output_caps_reconstruction_loss, squared_differences = 0, 0
            for this_shape in range(2):
                with tf.variable_scope('shape_' + str(this_shape)):

                    this_output_caps_reconstruction_loss, this_squared_differences = compute_reconstruction_loss(reconstruction_targets[:, :, :, this_shape], decoder_outputs[:, :, this_shape])
                    output_caps_reconstruction_loss = output_caps_reconstruction_loss + this_output_caps_reconstruction_loss
                    squared_differences = squared_differences + this_squared_differences

                tf.summary.scalar('reconstruction_loss_sum', output_caps_reconstruction_loss)

            # make an rgb tf.summary image. Note: there's sum fucked up dimension tweaking but it works. Don't worry about details.
            color_masks = np.array([[121, 199, 83],  # 0: vernier, green
                                    [220, 76, 70],  # 1: red
                                    [79, 132, 196]])  # 3: blue
            color_masks = np.expand_dims(color_masks, axis=1)
            color_masks = np.expand_dims(color_masks, axis=1)
            decoder_output_images = tf.reshape(decoder_outputs, [-1, im_size[0], im_size[1], simultaneous_shapes])
            decoder_output_images_rgb_0 = tf.image.grayscale_to_rgb(tf.expand_dims(decoder_output_images[:, :, :, 0], axis=-1)) * color_masks[0, :, :, :]
            decoder_output_images_rgb_1 = tf.image.grayscale_to_rgb(tf.expand_dims(decoder_output_images[:, :, :, 1], axis=-1)) * color_masks[1, :, :, :]
            decoder_output_images_sum = decoder_output_images_rgb_0 + decoder_output_images_rgb_1
            tf.summary.image('decoder_output', decoder_output_images_sum, 6)


    ####################################################################################################################
    # Final loss, accuracy, training operations, init & saver
    ####################################################################################################################


    with tf.name_scope('total_loss'):

        loss = tf.add_n([alpha_margin * margin_loss,
                         alpha_reconstruction * output_caps_reconstruction_loss,
                         alpha_vernier_offset * training_vernier_loss],
                        name="loss")
        tf.summary.scalar('total_loss', loss)

    with tf.name_scope('accuracy'):

        # we need to sort the labels by ascending order to make sure that we don't penalize the network when it guess,
        # e.g. classes [0, 2] but the labels are in fact [2, 0] for example.
        y_sorted = tf.contrib.framework.sort(y, axis=-1, direction='ASCENDING', name='y_sorted')
        y_pred_sorted = tf.contrib.framework.sort(y_pred, axis=-1, direction='ASCENDING', name='y_pred_sorted')
        correct = tf.equal(y_sorted, y_pred_sorted, name="correct")
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")
        tf.summary.scalar('accuracy', accuracy)

    # to write summaries during and prediction too
    eval_summary_hook = tf.train.SummarySaverHook(save_steps=25, output_dir=LOGDIR + '/eval', summary_op=tf.summary.merge_all())
    pred_summary_hook = tf.train.SummarySaverHook(save_steps=1, output_dir=LOGDIR + '/pred-' + str(time()), summary_op=tf.summary.merge_all())

    # Wrap all of this in an EstimatorSpec (cf. tf.Estimator tutorials, etc).
    if mode == tf.estimator.ModeKeys.PREDICT:
        # the following line is
        predictions = {'vernier_accuracy': tf.tile(tf.expand_dims(vernier_accuracy, -1), [batch_size])}
        spec = tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=predictions,
                                          prediction_hooks=[pred_summary_hook])    # to write summaries during prediction too)
        return spec

    else:

        # TRAINING OPERATIONS #
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        training_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step(), name="training_op")

        spec = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=training_op,
            eval_metric_ops={},
            evaluation_hooks=[eval_summary_hook])  # to write summaries during evaluatino too

        return spec
