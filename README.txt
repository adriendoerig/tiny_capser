TINY_CAPSER -- a tiny implementation of capser

DESCRIPTION:
Trains a capsule network (Sabour, Frosst & Hinton, 2017) on verniers+shapes, then evaluates the model on our lab's uncrowding stimuli (Manassi et al, 2016).

HARDWARE:
Will run on whatever it finds on your computer. CPU, GPU or multiple GPUs.

PREREQUISITS:
You will need to have installed tensorflow, matplotlib and scikit-learn.

MODEL AND DATASET PARAMETERS:
All parameters are set in parameters.py

DATASET:
Create datasets before running the model by running make_data.py
make_data.py creates .tfrecords files containing the data and saves them in ./data (cf. make_tf_dataset.py).
Nice tutorials about how to create .tfrecord files:
https://www.youtube.com/watch?v=oxrcZ9uUblI&authuser=0
https://www.youtube.com/watch?v=uIcqeP7MFH0&authuser=0

RUNNING THE MODEL:
To train and evaluate the model, run tiny_capser_main_script.py.
The model uses the tf.Estimator framework (quick explanation: https://www.youtube.com/watch?v=G7oolm0jU8I&authuser=0)
The model function is in tiny_capser_model_function.py
The input function is in tiny_capser_input_function.py
The training data is in ./data/train.tfrecords
The evaluation data in ./data/test_squares.tfrecords

OUTPUTS:
During training, lots of stuff is displayed in tensorboard (tensorflow's network visualization tool).
After training, look at decoded outputs and vernier decoder accuracy in tensorboard.

TENSORBOARD:
Open a terminal, cd to the data directory (cd data).
Run "tensorboard -logdir=./".
A web browser link should appear in the terminal. Use it in your favourite browser (if it doesn't work, try a different browser).
Tensorboard tutorial: https://www.youtube.com/watch?v=eBbEDRsCmv4&authuser=0