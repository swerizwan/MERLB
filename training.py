import tensorflow as tf
import numpy as np
import os
from pandas import read_csv
import pickle
tf.compat.v1.experimental.output_all_intermediates(True)
from models.modelsfile import early_fusion_model, late_fusion_model, deep_fusion_model, joint_fusion_model, merlb_fusion_model
from utils.utilsfile import calculate_weights_class, get_confusion_matrix
import keras.activations
from keras import regularizers

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/graphviz-2.38/release/bin/'


def labels_processing(input_raw_data):
    """This function accepts raw labels and organizes them into their respective 'head' categories. 
    It takes raw input data as a parameter and returns a tuple of arrays, where each array holds all 
    the labels associated with a specific output.
    """
    valence = np.array([input_raw_data['Valence_Negative'], input_raw_data['Valence_Neutral'], input_raw_data['Valence_Positive']])
    arousal = np.array([input_raw_data[' Arousal_Low'], input_raw_data[' Arousal_Neutral'], input_raw_data[' Arousal_High']])

    
    punching = input_raw_data[' Punching']
    respawning = input_raw_data[' Respawning']
    routing = input_raw_data[' Routing']
    exploring = input_raw_data[' Exploring']
    procurement = input_raw_data[' Procurement']
    fighting = input_raw_data[' Fighting']
    defeated = input_raw_data[' Defeated']
    defending = input_raw_data[' Defending']

    live_streaming = np.array([punching, respawning, routing, exploring, procurement, fighting, defeated, defending])
    return valence, arousal, live_streaming


def return_loaded_data(data_dictionary, labels, batch_size, flag_model, include_merlb_bias=False, shuffle_data=True):
    """Creates and provides a generator that dynamically fetches data from the specified directory when required.

Parameters:

data_dictionary: The directory containing folders with target data.
labels: Image width.
batch_size: Number of images in each batch.
flag_model: Desired model outputs.
include_merlb_bias: Number of images in each batch.
shuffle_data: Boolean to indicate whether data should be shuffled.
Returns:
A generator suitable for Keras models to read data.
    """
    input_data = read_csv(labels)
    if shuffle_data:
        input_data = input_data.sample(frac=1)
    index = 0
    while True:
        face_data = []
        audio_data = []
        live_streaming_data = []
        valence_labels = []
        arousal_labels = []
        live_streaming_labels = []
        current_batch_i = 0
        while current_batch_i < batch_size:
            if index >= len(input_data.index):
                index = 0
                if shuffle_data:
                    input_data = input_data.sample(frac=1)
            input_raw_data = input_data.iloc[index]
            processing_input_data = labels_processing(input_raw_data)

            valence_labels.append(processing_input_data[0])
            arousal_labels.append(processing_input_data[1])
            live_streaming_labels.append(processing_input_data[2])
            video_file = input_data.iloc[index]['File']
            face_data.append(np.load("%s/%s/face.npy" % (data_dictionary, video_file)))
            live_streaming_data.append(np.load("%s/%s/game.npy" % (data_dictionary, video_file)))
            audio_data.append(np.load("%s/%s/audio.npy" % (data_dictionary, video_file)))
            index += 1
            current_batch_i += 1

        if include_merlb_bias:
            inputs = [np.array(face_data), np.array(live_streaming_data), np.array(audio_data), np.ones((batch_size, 1))]
        else:
            inputs = [np.array(face_data), np.array(live_streaming_data), np.array(audio_data)]

        if flag_model == "both":
            outputs = [np.array(valence_labels), np.array(arousal_labels), np.array(live_streaming_labels)]
        elif flag_model == "emo":
            outputs = [np.array(valence_labels), np.array(arousal_labels)]
        else:
            outputs = [np.array(live_streaming_labels)]

        yield inputs, outputs


def loaded_data(data_dictionary, labels, batch_size, flag_model, include_merlb_bias=False):
    """Creates and provides a generator that fetches data from the specified directory on demand.

Parameters:
data_directory: Directory containing folders with target data.
labels: Width of the image.
batch_size: Number of images per batch.
include_merlb_bias: Whether to include a bias term.
Returns:
A generator suitable for use by a Keras model to access data, along with the total number of videos.
    """
    number_of_videos = len(read_csv(labels).index)
    return return_loaded_data(data_dictionary, labels, batch_size, flag_model, include_merlb_bias), number_of_videos

def experiment_executing(data_dictionary, time_steps, epochs, train_labels, test_labels, batch_size,
                   weights_class, flag_model, type_fusion, model_output, out_history, out_test):
    # Load data generators for training and testing
    train_generator, train_n = loaded_data(data_dictionary, train_labels, batch_size, flag_model, type_fusion == "merlb")
    test_generator, test_n = loaded_data(data_dictionary, test_labels, batch_size, flag_model, type_fusion == "merlb")

    # Create the appropriate fusion model based on type_fusion
    if type_fusion == "merlb":
        model = merlb_fusion_model(time_steps, flag_model)
    elif type_fusion == "late":
        model = late_fusion_model(time_steps, flag_model)
    elif type_fusion == "deep":
        model = deep_fusion_model(time_steps, flag_model)
    elif type_fusion == "joint":
        model = joint_fusion_model(time_steps, flag_model)
    else:
        model = early_fusion_model(time_steps, flag_model)

    # Train the model
    history = model.fit_generator(train_generator, steps_per_epoch=int(train_n / batch_size),
                                  epochs=epochs, class_weight=weights_class)

    # Save the training history and model weights
    with open('%s' % out_history, 'wb') as pi_file:
        pickle.dump(history.history, pi_file)
    model.save_weights(model_output)

    # Evaluate the model on the test set and save the confusion matrix
    get_confusion_matrix(model, test_generator, batch_size, test_n, flag_model, out_test)

def model_training(data_dictionary, time_steps, epochs, train_labels, test_labels, batch_size, model):
    # Calculate class weights for training data
    weights_valence, weights_arousal, weights_live_streaming = calculate_weights_class(train_labels)

    # Perform experiments for different fusion types
    experiment_executing(data_dictionary, time_steps, epochs, train_labels, test_labels, batch_size,
                   [weights_valence, weights_arousal, weights_live_streaming],
                   "both", model,
                   "trained_models/both_%s_fusion.h5" % model, 
                   "training_history/both_%s_fusion.p" % model,
                   "results/both_%s_fusion.txt" % model)

    experiment_executing(data_dictionary, time_steps, epochs, train_labels, test_labels, batch_size,
                   [weights_live_streaming],
                   "game", model,
                   "trained_models/game_%s_fusion.h5" % model, 
                   "training_history/game_%s_fusion.p" % model,
                   "results/game_%s_fusion.txt" % model)

    experiment_executing(data_dictionary, time_steps, epochs, train_labels, test_labels, batch_size,
                   [weights_valence, weights_arousal],
                   "emo", model,
                   "trained_models/emo_%s_fusion.h5" % model, 
                   "training_history/emo_%s_fusion.p" % model,
                   "results/emo_%s_fusion.txt" % model)

def main():
    # Create necessary directories if they don't exist
    if not os.path.exists("trained_models/"):
        os.makedirs("trained_models/")
    if not os.path.exists("training_history/"):
        os.makedirs("training_history/")
    if not os.path.exists("results/"):
        os.makedirs("results/")

    time_steps = 20
    batch_size = 32
    epochs = 100
    data_dictionary = "processed"
    train_labels = "trainingdata.csv"
    test_labels = "testingdata.csv"

    # Perform model training with different fusion types
    model_training(data_dictionary, time_steps, epochs, train_labels, test_labels, batch_size, "early")
    model_training(data_dictionary, time_steps, epochs, train_labels, test_labels, batch_size, "late")
    model_training(data_dictionary, time_steps, epochs, train_labels, test_labels, batch_size, "deep")
    model_training(data_dictionary, time_steps, epochs, train_labels, test_labels, batch_size, "joint")
    model_training(data_dictionary, time_steps, epochs, train_labels, test_labels, batch_size, "merlb")

if __name__ == "__main__":
    main()