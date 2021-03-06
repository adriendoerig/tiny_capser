from make_tf_dataset import *

# choose which sets to create
training = 1
testing_categories = 1
testing_individual_stimuli = 1

show_samples = 0

# create tfRecord data files
# training set
if training:
    make_multi_shape_tfRecords(stim_maker, shape_types, n_train_samples, train_data_path)
    if show_samples:
        show_data(train_data_path, 'multi_shape')

# test sets
if testing_categories:
    for category in test_stimuli.keys():
        make_config_tfRecords(stim_maker, test_stimuli[category], n_test_samples, os.path.join(data_path, "test_" + category + ".tfrecords"))
        if show_samples:
            show_data(os.path.join(data_path, "test_" + category + ".tfrecords"), 'config')

if testing_individual_stimuli:
    for category in test_stimuli.keys():
        stim_matrices = test_stimuli[category]
        for this_stim in range(3):
            curr_stim = stim_matrices[this_stim]
            if not os.path.exists(data_path+'/'+category):
                os.makedirs(data_path+'/'+category)
            make_config_tfRecords(stim_maker, [curr_stim], n_test_samples, os.path.join(data_path, category+"/test_" + category + str(this_stim) + ".tfrecords"))