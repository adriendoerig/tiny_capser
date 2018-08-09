import numpy as np
from parameters import *

# define a safe-norm to avoid infinities and zeros
def safe_norm(s, axis=-1, epsilon=1e-7, keep_dims=False, name=None):
    with tf.name_scope(name, default_name="safe_norm"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis, keep_dims=keep_dims)
        return tf.sqrt(squared_norm + epsilon)


# define the squash function (to apply to capsule vectors)
# a safe-norm is implemented to avoid 0 norms because they
# would fuck up the gradients etc.
def squash(s, axis=-1, epsilon=1e-7, name=None):
    with tf.name_scope(name, default_name="squash"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis, keep_dims=True)
        safe_norm_squash = tf.sqrt(squared_norm + epsilon)
        squash_factor = squared_norm / (1. + squared_norm)
        unit_vector = s / safe_norm_squash
        return squash_factor * unit_vector


# takes the first regular convolutional layers' output as input and creates the first capsules
# returns the flattened output of the primary capsule layer (only works to feed to a FC caps layer)
def primary_caps_layer(conv_output, caps1_n_maps, caps1_n_caps, caps1_n_dims,
                       conv_kernel_size, conv_strides, conv_padding='valid',
                       conv_activation=tf.nn.relu, print_shapes=False):
    with tf.name_scope('primary_capsules'):
        conv_params = {
            "filters": caps1_n_maps * caps1_n_dims,  # n convolutional filters
            "kernel_size": conv_kernel_size,
            "strides": conv_strides,
            "padding": conv_padding,
            "activation": conv_activation
        }

        # we will reshape this to create the capsules
        conv_for_caps = tf.layers.conv2d(conv_output, name="conv_for_caps", **conv_params)
        if print_shapes:
            print('shape of conv_for_caps: '+str(conv_for_caps))

        # in case we want to force the network to use a certain primary capsule map to represent certain shapes, we must
        # conserve all the map dimensions (we don't care about spatial position)
        caps_per_map = tf.cast(caps1_n_caps / caps1_n_maps, dtype=tf.int32, name='caps_per_map')
        caps1_raw_with_maps = tf.reshape(conv_for_caps, [batch_size, caps1_n_maps, caps_per_map, caps1_n_dims], name="caps1_raw_with_maps")

        # reshape the output to be caps1_n_dims-Dim capsules (since the next layer is FC, we don't need to
        # keep the [batch,xx,xx,n_feature_maps,caps1_n_dims] so we just flatten it to keep it simple)
        caps1_raw = tf.reshape(conv_for_caps, [batch_size, caps1_n_caps, caps1_n_dims], name="caps1_raw")
        # tf.summary.histogram('caps1_raw', caps1_raw)

        # squash capsule outputs
        caps1_output = squash(caps1_raw, name="caps1_output")
        caps1_output_with_maps = squash(caps1_raw_with_maps, name="caps1_output_with_maps")
        # tf.summary.histogram('caps1_output', caps1_output)
        if print_shapes:
            print('shape of caps1_output: '+str(caps1_output))
            print('shape of caps1_output_with_maps: ' + str(caps1_output_with_maps))

        return caps1_output, caps1_output_with_maps


# takes a (flattened) primary capsule layer caps1 output as input and creates a new fully connected capsule layer caps2
def primary_to_fc_caps_layer(input_batch, caps1_output, caps1_n_caps, caps1_n_dims, caps2_n_caps, caps2_n_dims,
                             rba_rounds=3, print_shapes=False):
    # Now the tricky part: create the predictions of secondary capsules' activities!
    # in essence, this is creating random weights from each dimension of each layer 1 capsule
    # to each dimension of each layer 2 capsule -- initializing random transforms on caps1 output vectors.
    # To make it efficient we use only tensorflow-friendly matrix multiplications. To this end,
    # we use tf.matmul, which performs element wise matrix in multidimensional arrays. To create
    # these arrays we use tf.tile a lot. See ageron github & video for more explanations.

    with tf.name_scope('primary_to_first_fc'):
        # initialise weights
        init_sigma = 0.01  # stdev of weights
        W_init = lambda: tf.random_normal(
            shape=(1, caps1_n_caps, caps2_n_caps, caps2_n_dims, caps1_n_dims),
            stddev=init_sigma, dtype=tf.float32, name="W_init")
        W = tf.Variable(W_init, dtype=tf.float32, name="W")

        # tile weights to [batch_size, caps1_n_caps, caps2_n_caps, caps2_n_dims, caps1_n_dims]
        # i.e. batch_size times a caps2_n_dims*caps1_n_dims array of [caps1_n_caps*caps2_n_caps] weight matrices
        # batch_size = tf.shape(input_batch)[0]  # note: tf.shape(X) is undefined until we fill the placeholder
        W_tiled = tf.tile(W, [batch_size, 1, 1, 1, 1], name="W_tiled")

        # tile caps1_output to [batch_size, caps1_n_caps, caps2_n_caps, caps2_n_dims, caps1_n_dims]
        # to do so, first we need to add the required dimensions with tf.expand_dims
        caps1_output_expanded = tf.expand_dims(caps1_output, -1,
                                               name="caps1_output_expanded")  # expand last dimension
        caps1_output_tile = tf.expand_dims(caps1_output_expanded, 2,
                                           name="caps1_output_tile")  # expand third dimension
        caps1_output_tiled = tf.tile(caps1_output_tile, [1, 1, caps2_n_caps, 1, 1],
                                     name="caps1_output_tiled")  # tile
        # check shapes
        if print_shapes:
            print('shape of tiled W: ' + str(W_tiled))
            print('shape of tiled caps1_output: ' + str(caps1_output_tiled))

        # Thanks to all this hard work, computing the secondary capsules' predicted activities is easy peasy:
        caps2_predicted = tf.matmul(W_tiled, caps1_output_tiled,
                                    name="caps2_predicted")

        # tf.summary.histogram('rba_0', caps2_predicted)

        # check shape
        if print_shapes:
            print('shape of caps2_predicted: ' + str(caps2_predicted))

        ################################################################################################################
        # ROUTING BY AGREEMENT iterative algorithm
        ################################################################################################################

        with tf.name_scope('routing_by_agreement'):

            def do_routing_cond(caps2_predicted, caps2_output, raw_weights, rba_iter, max_iter=rba_rounds):
                return tf.less(rba_iter, max_iter+1, name='do_routing_cond')

            def routing_by_agreement(caps2_predicted, caps2_output, raw_weights, rba_iter, max_iter=rba_rounds):

                # Round 1 of RbA

                # softmax on weights
                routing_weights = tf.nn.softmax(raw_weights, dim=2, name="routing_weights")

                # weighted sum of the lower layer predictions according to the routing weights
                weighted_predictions = tf.multiply(routing_weights, caps2_predicted,
                                                   name="weighted_predictions")
                weighted_sum = tf.reduce_sum(weighted_predictions, axis=1, keep_dims=True,
                                             name="weighted_sum")

                # squash
                caps2_rba_output = squash(weighted_sum, axis=-2,
                                          name="caps2_rba_output")

                # check shape
                if print_shapes:
                    print('shape of caps2_output after after RbA round: ' + str(caps2_rba_output))

                # to measure agreement, we just compute dot products between predictions and actual activations.
                # to do so, we will again use tf.matmul and tiling
                caps2_rba_output_tiled = tf.tile(
                    caps2_rba_output, [1, caps1_n_caps, 1, 1, 1],
                    name="caps2_rba_output_tiled")

                # check shape
                if print_shapes:
                    print('shape of TILED caps2_output after RbA round: ' + str(caps2_rba_output_tiled))

                # comput agreement is simple now
                agreement = tf.matmul(caps2_predicted, caps2_rba_output_tiled,
                                      transpose_a=True, name="agreement")

                # update routing weights based on agreement
                raw_weights_new = tf.add(raw_weights, agreement,
                                         name="raw_weights_round_new")

                return caps2_predicted, caps2_rba_output, raw_weights_new, tf.add(rba_iter, 1)

            # initialize routing weights
            raw_weights = tf.zeros([batch_size, caps1_n_caps, caps2_n_caps, 1, 1],
                                   dtype=np.float32, name="raw_weights")

            rba_iter = tf.constant(1, name='rba_iteration_counter')
            caps2_output = tf.zeros(shape=(batch_size, 1, caps2_n_caps, caps2_n_dims, 1), name='caps2_output')
            caps2_predicted, caps2_output, raw_weights, rba_iter = tf.while_loop(do_routing_cond, routing_by_agreement,
                                                                                 [caps2_predicted, caps2_output,
                                                                                  raw_weights, rba_iter])

        # This is the caps2 output!
        # tf.summary.histogram('rba_output', caps2_output)

        if print_shapes:
            print('shape of caps2_output after RbA termination: ' + str(caps2_output))

        return caps2_output



def caps_prediction(caps2_output, n_labels=1, print_shapes=False):
    with tf.name_scope('net_prediction'):

        y_proba = safe_norm(caps2_output, axis=-2, name="y_proba")

        if n_labels == 1:  # there is a single shape to classify
            y_proba_argmax = tf.argmax(y_proba, axis=2, name="y_proba")
            # get predicted class by squeezing out irrelevant dimensions
            y_pred = tf.squeeze(y_proba_argmax, axis=[1, 2], name="y_pred")
        else:  # there are more than one shape to classify
            _, y_pred = tf.nn.top_k(y_proba[:, 0, :, 0], 2, name="y_proba")
            y_pred = tf.cast(y_pred, tf.int64)  # need to cast for type compliance later)

        if print_shapes:
            # check shapes
            print('shape of prediction: '+str(y_pred))

        return y_pred


def compute_margin_loss(labels, caps2_output, caps2_n_caps, m_plus, m_minus, lambda_, print_shapes=False):
    with tf.name_scope('margin_loss'):

        if len(labels.shape) == 1:  # there is a single shape to classify

            # the T in the margin equation can be computed easily
            T = tf.one_hot(labels, depth=caps2_n_caps, name="T")
            if print_shapes:
                print('Computing output margin loss based on ONE label per image')
                print('shape of output margin loss function -- T: ' + str(T))

        else:  # there are more than one shape to classify

            # trick to get a vector for each image in the batch. Labels [0,2] -> [[1, 0, 1]] and [1,1] -> [[0, 1, 0]]
            T_raw = tf.one_hot(labels, depth=caps2_n_caps)
            T = tf.reduce_sum(T_raw, axis=1)
            T = tf.minimum(T, 1)
            if print_shapes:
                print('Computing output margin loss based on ' + str(len(labels.shape)) + ' labels per image')
                print('shape of output margin loss function -- T: ' + str(T))

        # the norms of the last capsules are taken as output probabilities
        caps2_output_norm = safe_norm(caps2_output, axis=-2, keep_dims=True, name="caps2_output_norm")
        if print_shapes:
            print('shape of output margin loss function -- caps2_output_norm: ' + str(caps2_output_norm))

        # present and absent errors go into the loss
        present_error_raw = tf.square(tf.maximum(0., m_plus - caps2_output_norm), name="present_error_raw")
        if print_shapes:
            print('shape of output margin loss function -- present_error_raw: ' + str(present_error_raw))
        present_error = tf.reshape(present_error_raw, shape=(batch_size, caps2_n_caps), name="present_error")  # there is a term for each of the caps2ncaps possible outputs
        if print_shapes:
            print('shape of output margin loss function -- present_error: ' + str(present_error))
        absent_error_raw = tf.square(tf.maximum(0., caps2_output_norm - m_minus), name="absent_error_raw")
        absent_error = tf.reshape(absent_error_raw, shape=(batch_size, caps2_n_caps), name="absent_error")    # there is a term for each of the caps2ncaps possible outputs

        # compute the margin loss
        L = tf.add(T * present_error, lambda_ * (1.0 - T) * absent_error, name="L")
        if print_shapes:
            print('shape of output margin loss function -- L: ' + str(L))
        margin_loss = tf.reduce_mean(tf.reduce_sum(L, axis=1), name="margin_loss")
        # tf.summary.scalar('margin_loss_output', margin_loss)

        return margin_loss


def create_masked_decoder_input(labels, labels_pred, caps_output, n_caps, caps_n_dims, mask_with_labels,
                                print_shapes=False):
    # CREATE MASK #

    with tf.name_scope('create_masked_decoder_input'):
        # use the above condition to find out which (label vs. predicted label) to use. returns, for example, 3.
        reconstruction_targets = tf.cond(mask_with_labels,  # condition
                                         lambda: labels,  # if True
                                         lambda: labels_pred,  # if False
                                         name="reconstruction_targets")

        # Let's create the reconstruction mask. It should be equal to 1.0 for the target class, and 0.0 for the
        # other classes, for each instance in the batch.
        reconstruction_mask = tf.one_hot(reconstruction_targets, depth=n_caps, name="reconstruction_mask")
        if len(labels.shape) > 1:  # there are different shapes in the image
            reconstruction_mask = tf.reduce_sum(reconstruction_mask, axis=1)
            reconstruction_mask = tf.minimum(reconstruction_mask, 1)

        # caps2_output shape is (batch size, 1, 10, 16, 1). We want to multiply it by the reconstruction_mask,
        # but the shape of the reconstruction_mask is (batch size, 10). We must reshape it to (batch size, 1, 10, 1, 1)
        # to make multiplication possible:
        reconstruction_mask_reshaped = tf.reshape(reconstruction_mask, [batch_size, 1, n_caps, 1, 1], name="reconstruction_mask_reshaped")

        # Apply the mask!
        caps2_output_masked = tf.multiply(caps_output, reconstruction_mask_reshaped, name="caps2_output_masked")

        if print_shapes:
            # check shape
            print('shape of masked output: ' + str(caps2_output_masked))

        # flatten the masked output to feed to the decoder
        decoder_input = tf.reshape(caps2_output_masked, [batch_size, n_caps * caps_n_dims], name="decoder_input")

        if print_shapes:
            # check shape
            print('shape of decoder input: ' + str(decoder_input))

        return decoder_input


def compute_vernier_offset_loss(vernier_capsule, labels, print_shapes=False):

    with tf.name_scope('vernier_offset_loss'):

        if vernier_label_encoding is 'lr_01':
            depth = 2
        elif vernier_label_encoding is 'nothinlr_012':
            depth = 3

        one_hot_offsets = tf.one_hot(tf.cast(labels, tf.int32), depth)
        offset_logits = tf.layers.dense(vernier_capsule, depth, activation=tf.nn.relu, name="offset_logits")
        offset_xent = tf.losses.softmax_cross_entropy(one_hot_offsets, offset_logits)
        tf.summary.scalar('training_vernier_offset_xentropy', offset_xent)
        correct = tf.equal(labels, tf.cast(tf.argmax(offset_logits, axis=1), tf.float32), name="correct")
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")
        tf.summary.scalar('accuracy', accuracy)

        if print_shapes:
            print('shape of compute_vernier_offset_loss -- input to decoder: ' + str(vernier_capsule))
            print('shape of compute_vernier_offset_loss -- one_hot_offsets: ' + str(one_hot_offsets))
            print('shape of compute_vernier_offset_loss -- offset_logits: ' + str(offset_logits))
            print('shape of compute_vernier_offset_loss -- offset_xent: ' + str(offset_xent))

        return offset_xent, accuracy, offset_logits


def decoder_with_mask_batch_norm(decoder_input, n_output, n_hidden1=None, n_hidden2=None, n_hidden3=None, phase=True, name=''):

    with tf.name_scope(name+"decoder"):

        if n_hidden1 is not None:
            hidden1 = tf.layers.dense(decoder_input, n_hidden1, use_bias=False, name=name+'hidden1', activation=None)
            hidden1 = tf.layers.batch_normalization(hidden1, training=phase, name=name+'hidden1_bn')
            hidden1 = decoder_activation_function(hidden1, name='hidden1_activation')
            tf.summary.histogram(name+'_hidden1_bn', hidden1)
            if n_hidden2 is not None:
                hidden2 = tf.layers.dense(hidden1, n_hidden2, use_bias=False, name=name + 'hidden2', activation=None)
                hidden2 = tf.layers.batch_normalization(hidden2, training=phase, name=name + 'hidden2_bn')
                hidden2 = decoder_activation_function(hidden2, name='hidden2_activation')
                tf.summary.histogram(name+'_hidden2_bn', hidden2)
                if n_hidden3 is not None:
                    hidden3 = tf.layers.dense(hidden2, n_hidden3, use_bias=False, name=name + 'hidden3', activation=None)
                    hidden3 = tf.layers.batch_normalization(hidden3, training=phase, name=name + 'hidden3_bn')
                    hidden3 = decoder_activation_function(hidden3, name='hidden3_activation')
                    tf.summary.histogram(name + '_hidden3_bn', hidden3)
                    decoder_output = tf.layers.dense(hidden3, n_output, activation=tf.nn.sigmoid, name=name + "_output")
                else:
                    decoder_output = tf.layers.dense(hidden2, n_output, activation=tf.nn.sigmoid, name=name+"_output")
            else:
                decoder_output = tf.layers.dense(hidden1, n_output, activation=tf.nn.sigmoid, name=name + "_output")
        else:
            decoder_output = tf.layers.dense(decoder_input, n_output, activation=tf.nn.sigmoid, name=name + "_output")

        tf.summary.histogram(name+'_output', decoder_output)

        return decoder_output


# takes a n*n input and a flat decoder output
def compute_reconstruction_loss(input, reconstruction):

    with tf.name_scope('reconstruction_loss'):

        X_flat = tf.reshape(input, [batch_size, tf.shape(reconstruction)[1]], name="X_flat")
        X_flat = tf.cast(X_flat, tf.float32)  # converts to type tf.float32

        squared_difference = tf.square(X_flat - reconstruction, name="squared_difference")
        reconstruction_loss = tf.reduce_sum(squared_difference,  name="reconstruction_loss")
        stimuli_square_differences = tf.reduce_sum(squared_difference, axis=1, name="square_diff_for_each_stimulus")

        return reconstruction_loss, stimuli_square_differences
