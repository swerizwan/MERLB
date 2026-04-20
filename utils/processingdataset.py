import cv2
import numpy as np
import os
import subprocess
from librosa import load
from scipy.stats import zscore
import math
import json
from random import shuffle
from pandas import read_csv

def generated_value_face_loc(id_streamer):
	"""Obtains the bounding box of the webcam for a specified streamer.
	Input: Streamer's ID
	Output: Bounding box represented as a tuple.
	"""
	with open("webcam_boxes.json", 'r') as f:
		bounding_boxes = json.load(f)
		return bounding_boxes[id_streamer]


def generated_value_id_streamer(string_video):
	"""Obtains the streamer's ID using the video's name as input.
	Input: A string representing the video's name.
	Output: A string that holds the streamer's ID.
	"""
	id_streamer = ""
	started = False
	for char in string_video:
		if char == '_': 
			if not started:
				started = True
			else:
				return id_streamer
		elif started:
			id_streamer += char


def frames_processing(video, dictionary_output, face_loc):
	"""Captures and stores frames from the given video as an npy file.

	Input video is processed.
	Frames are saved to the specified destination.
	Webcam bounding box information is included.
	"""
	cap = cv2.VideoCapture(video)
	images_face = []
	images_live_streaming = []

	if not os.path.exists(dictionary_output):
		os.makedirs(dictionary_output)

	frames_n = 20
	frame_current = 0

	if not os.path.exists(dictionary_output):
		os.makedirs(dictionary_output)

	while True:
		if frame_current >= frames_n:
			break
		frame_current += 1
		_, frame = cap.read()
		face = np.array(frame[face_loc[1]:face_loc[3], face_loc[0]:face_loc[2], :])
		cv2.rectangle(frame, (face_loc[0], face_loc[1]), (face_loc[2], face_loc[3]), -1, -1)

		face = cv2.resize(face, (64, 64))
		live_streaming = cv2.resize(frame, (128, 128))
		images_face.append(face)
		images_live_streaming.append(live_streaming)

	processed_face = np.array(images_face)/255
	np.save("%s/face.npy" % dictionary_output, processed_face)

	processed_live_streaming = np.array(images_live_streaming)/255
	np.save("%s/game.npy" % dictionary_output, processed_live_streaming)


def audio_processing(video, dictionary_output):
	"""Converts and stores the audio from the given video as an npy file.

	video: The input video.
	dictionary_output: The destination to store the audio data.
	"""
	if not os.path.exists(dictionary_output):
		os.makedirs(dictionary_output)
	os.system("ffmpeg -hide_banner -loglevel panic -y -i %s -vn -acodec copy %s" % (video, dictionary_output + "/raw_audio.mp4"))
	wav_value, rated_value = load(dictionary_output + "/raw_audio.mp4")
	rated_value_pf = int(rated_value/4)
	wav_value = zscore(wav_value)
	slices_audio = []
	for i in range(0, 20):
		start = i*rated_value_pf
		end = i*rated_value_pf+rated_value_pf
		if end < len(wav_value):
			frame_audio = wav_value[start:end]
			slices_audio.append(frame_audio)
		else:
			frame_audio = wav_value[start:len(wav_value)]
			missed = end - len(wav_value)
			frame_audio = np.concatenate((frame_audio, np.zeros((missed,)).astype("float32")), axis=0)
			slices_audio.append(frame_audio)
	array_processing = np.expand_dims(np.array(slices_audio), axis=2)
	np.save("%s/audio.npy" % dictionary_output, array_processing)


def main():
	data_dictionary = "data/"
	output_dir = "processed/"
	videos = [f for f in os.listdir(data_dictionary) if f.endswith(".mp4")]
	
	for video in videos:
		id_streamer = generated_value_id_streamer(video)
		frames_processing(data_dictionary + video, output_dir + video, generated_value_face_loc(id_streamer))
		audio_processing(data_dictionary + video, output_dir + video)


if __name__ == "__main__":
	main()
