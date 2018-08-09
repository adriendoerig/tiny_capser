import tensorflow as tf
import os
import random

### data  ###

data_path = './data'                                    # save your tfRecord data files here

# training set
train_data_path = data_path+'/train.tfrecords'          # where the training data file is located
test_data_path = data_path+'/test_squares.tfrecords'    # where the testing data file is located
n_train_samples = 1000                                  # number of different stimuli in an epoch
batch_size = 64                                         # stimuli per batch
buffer_size = 1024                                      # number of stimuli simultaneously in memory (I think).
n_epochs = 50                                           # number of epochs
n_steps = n_train_samples*n_epochs/batch_size           # number of training steps

# testing sets
n_test_samples = 100                                                                # number of stimuli for each testing condition
test_stimuli = {'squares': [None, [[1]], [[1, 1, 1, 1, 1]]]}                        # see batchMaker.py if you are interested in how to sreate the stimuli
test_filenames = [data_path+'/test_'+keys+'.tfrecords' for keys in test_stimuli]    # filenames of the .tfRecords files


### stimulus params - APPLIED WHEN YOU RUN MAKE_DATA.PY, NOT WHEN YOU RUN THE MAIN SCRIPT ###

im_size = (45, 100)                             # size of full image
shape_size = 18                                 # size of a single shape in pixels
random_size = True                              # shape_size will vary around shape_size
random_pixels = .4                              # stimulus pixels are drawn from random.uniform(1-random_pixels,1+random_pixels). So use 0 for deterministic stimuli. see batchMaker.py
simultaneous_shapes = 2                         # number of different shapes in an image. NOTE: more than 2 is not supported at the moment
bar_width = 1                                   # thickness of elements' bars
noise_level = 0.0                               # add noise
shape_types = [0, 1, 2, 3, 4, 5, 6, 9]          # see batchMaker.drawShape for number-shape correspondences
group_last_shapes = 1                           # attributes the same label to the last n shapeTypes
fixed_stim_position = None                      # put top left corner of all stimuli at fixed_position
normalize_images = False                        # make each image mean=0, std=1
vernier_normalization_exp = 0                   # to give more importance to the vernier (see batchMaker). Use 0 for no effect. > 0  -> favour vernier during training
normalize_sets = False                          # compute mean and std over 100 images and use this estimate to normalize each image
max_rows, max_cols = 1, 5                       # max number of rows, columns of shape grids
vernier_grids = False                           # if true, verniers come in grids like other shapes. Only single verniers otherwise.

label_to_shape = {0: 'vernier', 1: 'squares', 2: 'circles', 4: 'hexagons', 5: 'octagons', 6: '4stars', 7: '7stars', 8: 'stuff'}
shape_to_label = dict([[v, k] for k, v in label_to_shape.items()])


### network params ###

# learning rate
learning_rate = .0005

# conv layers
conv_activation_function = tf.nn.elu
conv1_params = {"filters": 32,
                "kernel_size": 7,
                "strides": 1,
                "padding": "valid",
                "activation": conv_activation_function,
                }
conv2_params = {"filters": 32,
                "kernel_size": 7,
                "strides": 2,
                "padding": "valid",
                "activation": conv_activation_function,
                }

# primary capsules
caps1_n_maps = len(label_to_shape)  # number of capsules at level 1 of capsules
caps1_n_dims = 16                   # number of dimension per capsule
conv_caps_params = {"filters": caps1_n_maps * caps1_n_dims,
                    "kernel_size": 6,
                    "strides": 2,
                    "padding": "valid",
                    "activation": conv_activation_function,
                    }

# output capsules
caps2_n_caps = len(label_to_shape)  # number of capsules
caps2_n_dims = 16                   # of n dimensions
rba_rounds = 2

# margin loss parameters. Cf. aurélien géron's video
alpha_margin = 3.333
m_plus = .9
m_minus = .1
lambda_ = .5

# loss on a decoder trying to determine vernier orientation from the vernier output capsule
vernier_offset_loss = True
vernier_label_encoding = 'lr_01'        # 'lr_01' or 'nothinglr_012'
alpha_vernier_offset = 1                # vernier loss multiplier. needed to match the magnitude of other losses

# reconstruction loss multiplier. needed to match the magnitude of other losses
alpha_reconstruction = .0005

# image decoder parameters
decoder_activation_function = tf.nn.elu
output_caps_decoder_n_hidden1 = 512
output_caps_decoder_n_hidden2 = 1024
output_caps_decoder_n_output = im_size[0] * im_size[1]


### directories ###
MODEL_NAME = 'BS_'+str(batch_size)+'_C1DIM_'+str(caps1_n_dims)+'_C2DIM_'+str(caps2_n_dims)+'_LR_'+str(learning_rate)
LOGDIR = data_path + '/' + MODEL_NAME + '/'  # will be redefined below
if not os.path.exists(LOGDIR):
    os.makedirs(LOGDIR)
