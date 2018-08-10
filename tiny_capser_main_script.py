from tiny_capser_model_fn import model_fn
from tiny_capser_input_fn import *
import logging

# reproducibility
tf.reset_default_graph()
tf.set_random_seed(42)

# directory management
print('###################################################################################################')
print('################################### WELCOME, THIS IS TINY_CAPSER ##################################')
print('###################################################################################################')

# create estimator for model (the model is described in capser_7_model_fn)
capser = tf.estimator.Estimator(model_fn=model_fn, params={'model_batch_size': batch_size}, model_dir=LOGDIR)

# to output loss in the terminal every few steps
logging.getLogger().setLevel(logging.INFO)  # to show info about training progress

# tell the estimator where to get training and eval data, and for how long to train
train_spec = tf.estimator.TrainSpec(train_input_fn, max_steps=2000)
eval_spec = tf.estimator.EvalSpec(lambda: input_fn_config(data_path+'/test_squares.tfrecords'), steps=100)

# train (and evaluate from time to time)!
tf.estimator.train_and_evaluate(capser, train_spec, eval_spec)
