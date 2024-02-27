from models.basemodelsfile import time_distributed_netface, time_distributed_netlivestreaming, time_distributed_netaudio
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, BatchNormalization, Concatenate, Lambda, Dot, Reshape
from tensorflow.keras.callbacks import Callback,ModelCheckpoint
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.regularizers import l2
import os
from models.extern.MERLBLayersfile import MERLB_Layer
import keras.activations
from keras import regularizers

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/graphviz-2.38/release/bin/'

def get_f1_score(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val


def early_fusion_model(time_steps, flag_model):
	"""Builds and returns a full early fusion model

    :param time_steps: The length of the video time series sequence  
    :param flag_model: Which outputs the model should use ("both", "game", "emo")
    :return: The model (as a Keras model)
    """
	# Define input layers for each modality
	input_face = Input((time_steps, 64, 64, 3))
	input_live_streaming = Input((time_steps, 128, 128, 3))
	input_audio = Input((time_steps, 5512, 1))

	# Apply time-distributed networks to extract features
	features_face = time_distributed_netface((64, 64, 3), "face_")(input_face)
	features_live_streaming = time_distributed_netlivestreaming((128, 128, 3), "game_")(input_live_streaming)
	features_audio = time_distributed_netaudio((5512, 1), 512)(input_audio)

	# Concatenate features from all modalities
	feats_hidden = Concatenate()([features_face, features_live_streaming, features_audio])
	feats_hidden = BatchNormalization()(feats_hidden)
	feats_hidden = Dropout(0.2)(feats_hidden)

	# Define LSTM layers to capture temporal dependencies
	feats_hidden = LSTM(384, return_sequences=True)(feats_hidden)
	feats_hidden = LSTM(384, return_sequences=False)(feats_hidden)
	feats_hidden = BatchNormalization()(feats_hidden)
	feats_hidden = Dropout(0.2)(feats_hidden)

	# Output layers
	outputs = []

	if flag_model == "both" or flag_model == "emo":
		feats_valence = Dense(128, activation="relu")(feats_hidden)
		valence = Dense(3, activation='softmax')(feats_valence)
		outputs.append(valence)

		feats_arousal = Dense(128, activation="relu")(feats_hidden)
		arousal = Dense(3, activation='softmax')(feats_arousal)
		outputs.append(arousal)

	if flag_model == "both" or flag_model == "game":
		feats_live_streaming = Dense(128, activation="relu")(feats_hidden)
		live_streaming = Dense(8, activation='softmax')(feats_live_streaming)
		outputs.append(live_streaming)

	# Define the model
	model = Model([input_face, input_live_streaming, input_audio], outputs)

	# Compile the model
	opt = Adam(lr=0.0005)
	plot_model(model, to_file='model_imgs/%s_early.png' % flag_model, show_shapes=True)
	model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy', get_f1_score])
	print(model.summary())
	return model


def late_fusion_model(time_steps, flag_model):
	"""Builds and returns a full late fusion model

    :param time_steps: The length of the video time series sequence  
    :param flag_model: Which outputs the model should use ("both", "game", "emo")
    :return: The model (as a Keras model)
    """
	# Define input layers for each modality
	input_face = Input((time_steps, 64, 64, 3))
	input_live_streaming = Input((time_steps, 128, 128, 3))
	input_audio = Input((time_steps, 5512, 1))

	# Apply time-distributed networks to extract features
	features_face = time_distributed_netface((64, 64, 3), "face_")(input_face)
	features_live_streaming = time_distributed_netlivestreaming((128, 128, 3), "game_")(input_live_streaming)
	features_audio = time_distributed_netaudio((5512, 1), 512)(input_audio)

	# Batch normalization and dropout for each modality's features
	features_face = BatchNormalization()(features_face)
	features_face = Dropout(0.2)(features_face)
	features_face = LSTM(128, return_sequences=True)(features_face)
	features_face = LSTM(128, return_sequences=False)(features_face)

	features_live_streaming = BatchNormalization()(features_live_streaming)
	features_live_streaming = Dropout(0.2)(features_live_streaming)
	features_live_streaming = LSTM(128, return_sequences=True)(features_live_streaming)
	features_live_streaming = LSTM(128, return_sequences=False)(features_live_streaming)

	# Define LSTM layers to capture temporal dependencies
	features_audio = BatchNormalization()(features_audio)
	features_audio = Dropout(0.2)(features_audio)
	features_audio = LSTM(128, return_sequences=True)(features_audio)
	features_audio = LSTM(128, return_sequences=False)(features_audio)

	feats_hidden = Concatenate()([features_face, features_live_streaming, features_audio])
	feats_hidden = BatchNormalization()(feats_hidden)
	feats_hidden = Dropout(0.2)(feats_hidden)

	# Output layers
	outputs = []

	if flag_model == "both" or flag_model == "emo":
		feats_valence = Dense(128, activation="relu")(feats_hidden)
		valence = Dense(3, activation='softmax')(feats_valence)
		outputs.append(valence)

		feats_arousal = Dense(128, activation="relu")(feats_hidden)
		arousal = Dense(3, activation='softmax')(feats_arousal)
		outputs.append(arousal)

	if flag_model == "both" or flag_model == "game":
		feats_live_streaming = Dense(128, activation="relu")(feats_hidden)
		live_streaming = Dense(8, activation='softmax')(feats_live_streaming)
		outputs.append(live_streaming)

	# Define the model
	model = Model([input_face, input_live_streaming, input_audio], outputs)

	# Compile the model
	opt = Adam(lr=0.0005)
	plot_model(model, to_file='model_imgs/%s_late.png' % flag_model, show_shapes=True)
	model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy', get_f1_score])
	print(model.summary())
	return model

def deep_fusion_model(time_steps, flag_model):
	"""Builds and returns a Deep Fusion Network model

	:param time_steps: The length of the input time series sequence  
	:return: The model (as a Keras model)
	"""
	# Define input layers for each modality
	input_face = Input((time_steps, 64, 64, 3))
	input_live_streaming = Input((time_steps, 128, 128, 3))
	input_audio = Input((time_steps, 5512, 1))

	# Apply time-distributed networks to extract features
	features_face = time_distributed_network_face((64, 64, 3), "face_")(input_face)
	features_live_streaming = time_distributed_network_live_streaming((128, 128, 3), "game_")(input_live_streaming)
	features_audio = time_distributed_network_audio((5512, 1), 512)(input_audio)

	# Batch normalization and dropout for each modality's features
	features_face = BatchNormalization()(features_face)
	features_face = Dropout(0.2)(features_face)
	features_live_streaming = BatchNormalization()(features_live_streaming)
	features_live_streaming = Dropout(0.2)(features_live_streaming)
	features_audio = BatchNormalization()(features_audio)
	features_audio = Dropout(0.2)(features_audio)

	# Define LSTM layers to capture temporal dependencies
	features_face = LSTM(128, return_sequences=True)(features_face)
	features_face = LSTM(128, return_sequences=False)(features_face)
	features_live_streaming = LSTM(128, return_sequences=True)(features_live_streaming)
	features_live_streaming = LSTM(128, return_sequences=False)(features_live_streaming)
	features_audio = LSTM(128, return_sequences=True)(features_audio)
	features_audio = LSTM(128, return_sequences=False)(features_audio)

	# Concatenate features from all modalities
	concatenated_features = Concatenate()([features_face, features_live_streaming, features_audio])

	# Batch normalization and dropout for concatenated features
	feats_hidden = BatchNormalization()(concatenated_features)
	feats_hidden = Dropout(0.2)(feats_hidden)

	# Output layers
	outputs = []

	if flag_model == "both" or flag_model == "emo":
		feats_valence = Dense(128, activation="relu")(feats_hidden)
		valence = Dense(3, activation='softmax')(feats_valence)
		outputs.append(valence)

		feats_arousal = Dense(128, activation="relu")(feats_hidden)
		arousal = Dense(3, activation='softmax')(feats_arousal)
		outputs.append(arousal)

	if flag_model == "both" or flag_model == "game":
		feats_live_streaming = Dense(128, activation="relu")(feats_hidden)
		live_streaming = Dense(8, activation='softmax')(feats_live_streaming)
		outputs.append(live_streaming)

	# Define the model
	model = Model(inputs=[input_face, input_live_streaming, input_audio],
				outputs=[output_valence, output_arousal, output_live_streaming])

	# Compile the model
	optimizer = Adam(lr=0.0005)
	plot_model(model, to_file='model_imgs/%s_deep.png' % flag_model, show_shapes=True)
	model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy', get_f1_score])
	print(model.summary())
	return model

def joint_fusion_model(time_steps, flag_model):
	"""Builds and returns a joint fusion network model

	:param time_steps: The length of the input time series sequence  
	:return: The model (as a Keras model)
	"""
	# Define input layers for each modality
	input_face = Input((time_steps, 64, 64, 3), name='input_face')
	input_live_streaming = Input((time_steps, 128, 128, 3), name='input_live_streaming')
	input_audio = Input((time_steps, 5512, 1), name='input_audio')

	# Apply time-distributed networks to extract features
	features_face = time_distributed_netface((64, 64, 3), "face_")(input_face)
	features_live_streaming = time_distributed_netlivestreaming((128, 128, 3), "game_")(input_live_streaming)
	features_audio = time_distributed_netaudio((5512, 1), 512)(input_audio)

	# Batch normalization and dropout for each modality's features
	features_face = BatchNormalization()(features_face)
	features_face = Dropout(0.2)(features_face)
	features_live_streaming = BatchNormalization()(features_live_streaming)
	features_live_streaming = Dropout(0.2)(features_live_streaming)
	features_audio = BatchNormalization()(features_audio)
	features_audio = Dropout(0.2)(features_audio)

	# Define LSTM layers to capture temporal dependencies
	features_face = LSTM(128, return_sequences=True)(features_face)
	features_face = LSTM(128, return_sequences=False)(features_face)
	features_live_streaming = LSTM(128, return_sequences=True)(features_live_streaming)
	features_live_streaming = LSTM(128, return_sequences=False)(features_live_streaming)
	features_audio = LSTM(128, return_sequences=True)(features_audio)
	features_audio = LSTM(128, return_sequences=False)(features_audio)

	# Concatenate features from all modalities
	concatenated_features = Concatenate(name='concatenated_features')([features_face, features_live_streaming, features_audio])

	# Batch normalization and dropout for concatenated features
	feats_hidden = BatchNormalization()(concatenated_features)
	feats_hidden = Dropout(0.2)(feats_hidden)

	# Output layers
	outputs = []

	if flag_model == "both" or flag_model == "emo":
		feats_valence = Dense(128, activation="relu")(feats_hidden)
		valence = Dense(3, activation='softmax')(feats_valence)
		outputs.append(valence)

		feats_arousal = Dense(128, activation="relu")(feats_hidden)
		arousal = Dense(3, activation='softmax')(feats_arousal)
		outputs.append(arousal)

	if flag_model == "both" or flag_model == "game":
		feats_live_streaming = Dense(128, activation="relu")(feats_hidden)
		live_streaming = Dense(8, activation='softmax')(feats_live_streaming)
		outputs.append(live_streaming)

	# Define the model
	model = Model(inputs=[input_face, input_live_streaming, input_audio],
				outputs=[output_valence, output_arousal, output_live_streaming])

	# Compile the model
	optimizer = Adam(lr=0.0005)
	plot_model(model, to_file='model_imgs/%s_deep.png' % flag_model, show_shapes=True)
	model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy', get_f1_score])
	print(model.summary())
	return model

def merlb_fusion_model(time_steps, flag_model):
	"""Builds and returns a full merlb fusion model

    :param time_steps: The length of the video time series sequence  
    :param flag_model: Which outputs the model should use ("both", "game", "emo")
    :return: The model (as a Keras model)
    """
	# Define input layers for each modality
	input_face = Input((time_steps, 64, 64, 3))
	input_live_streaming = Input((time_steps, 128, 128, 3))
	input_audio = Input((time_steps, 5512, 1))
	bias = Input((1,))

	# Apply time-distributed networks to extract features
	features_face = time_distributed_netface((64, 64, 3), "face_")(input_face)
	features_live_streaming = time_distributed_netlivestreaming((128, 128, 3), "game_")(input_live_streaming)
	features_audio = time_distributed_netaudio((5512, 1), 512)(input_audio)

	# Batch normalization and dropout for each modality's features
	features_face = BatchNormalization()(features_face)
	features_face = Dropout(0.2)(features_face)
	features_face = LSTM(128, return_sequences=True)(features_face)
	features_face = LSTM(128, return_sequences=False)(features_face)

	features_live_streaming = BatchNormalization()(features_live_streaming)
	features_live_streaming = Dropout(0.2)(features_live_streaming)
	features_live_streaming = LSTM(128, return_sequences=True)(features_live_streaming)
	features_live_streaming = LSTM(128, return_sequences=False)(features_live_streaming)

	# Define LSTM layers to capture temporal dependencies
	features_audio = BatchNormalization()(features_audio)
	features_audio = Dropout(0.2)(features_audio)
	features_audio = LSTM(128, return_sequences=True)(features_audio)
	features_audio = LSTM(128, return_sequences=False)(features_audio)

	reshape_1 = Reshape((1, 129))(Concatenate()([bias, features_face]))
	reshape_2 = Reshape((1, 129))(Concatenate()([bias, features_live_streaming]))
	reshape_3 = Reshape((1, 129))(Concatenate()([bias, features_audio]))

	x = Dot(axes=1)([reshape_1, reshape_2])
	x = Reshape((1, 129 * 129))(x)
	x = Dot(axes=1)([x, reshape_3])
	feats_hidden = Reshape((129, 129, 129))(x)

	print("Tensor Shape: ", feats_hidden.shape)
	feats_hidden = MERLB_Layer(list_shape_input=[3, 43, 129, 43, 3],
	                        list_shape_output=[2, 4, 4, 4, 3],
	                        list_ranks=[1, 2, 4, 4, 2, 1],
	                        activation='relu', initializer_kernel=keras.regularizers.l2(5e-4), dtype=feats_hidden.dtype, debug=False)(feats_hidden)
	print("After MERLB Shape: ", feats_hidden.shape)

	feats_hidden = BatchNormalization()(feats_hidden)
	feats_hidden = Dropout(0.2)(feats_hidden)

	# Output layers
	outputs = []

	if flag_model == "both" or flag_model == "emo":
		feats_valence = Dense(128, activation="relu")(feats_hidden)
		valence = Dense(3, activation='softmax')(feats_valence)
		outputs.append(valence)

		feats_arousal = Dense(128, activation="relu")(feats_hidden)
		arousal = Dense(3, activation='softmax')(feats_arousal)
		outputs.append(arousal)

	if flag_model == "both" or flag_model == "game":
		feats_live_streaming = Dense(128, activation="relu")(feats_hidden)
		live_streaming = Dense(8, activation='softmax')(feats_live_streaming)
		outputs.append(live_streaming)

	# Define the model
	model = Model([input_face, input_live_streaming, input_audio, bias], outputs)

	# Compile the model
	opt = Adam(lr=0.0005)
	plot_model(model, to_file='model_imgs/%s_merlb.png' % flag_model, show_shapes=True)
	model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy', get_f1_score])
	print(model.summary())
	return model