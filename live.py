from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import cv2
import os
import time
import argparse
import time

import numpy as np
import tensorflow as tf

# Disable eager execution for compatibility with TensorFlow 2.x
tf.compat.v1.disable_eager_execution()

# Function to load a TensorFlow graph from a model file
def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.compat.v1.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph

# Function to read and preprocess an image file for inference
def read_tensor_from_image_file(file_name, input_height=299, input_width=299,
                                input_mean=0, input_std=255):
    input_name = "file_reader"
    output_name = "normalized"
    file_reader = tf.compat.v1.read_file(file_name, input_name)
    
    # Decode different image formats based on file extension
    if file_name.endswith(".png"):
        image_reader = tf.image.decode_png(file_reader, channels=3, name='png_reader')
    elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(tf.image.decode_gif(file_reader, name='gif_reader'))
    elif file_name.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
    else:
        image_reader = tf.image.decode_jpeg(file_reader, channels=3, name='jpeg_reader')

    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.compat.v1.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.compat.v1.Session()
    result = sess.run(normalized)

    return result

# Function to load labels from a file
def load_labels(label_file):
    label = []

    proto_as_ascii_lines = tf.compat.v1.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())

    return label

# Main function for performing image classification
def main(img):
    file_name = img
    model_file = "models/saved/retrained_graph.pb"
    label_file = "models/saved/retrained_labels.txt"
    input_height = 224
    input_width = 224
    input_mean = 128
    input_std = 128
    input_layer = "input"
    output_layer = "final_result"

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="image to be processed")
    parser.add_argument("--graph", help="graph/model to be executed")
    parser.add_argument("--labels", help="name of file containing labels")
    parser.add_argument("--input_height", type=int, help="input height")
    parser.add_argument("--input_width", type=int, help="input width")
    parser.add_argument("--input_mean", type=int, help="input mean")
    parser.add_argument("--input_std", type=int, help="input std")
    parser.add_argument("--input_layer", help="name of input layer")
    parser.add_argument("--output_layer", help="name of output layer")
    args = parser.parse_args()

    # Override default parameters with command line arguments if provided
    if args.graph:
        model_file = args.graph
    if args.image:
        file_name = args.image
    if args.labels:
        label_file = args.labels
    if args.input_height:
        input_height = args.input_height
    if args.input_width:
        input_width = args.input_width
    if args.input_mean:
        input_mean = args.input_mean
    if args.input_std:
        input_std = args.input_std
    if args.input_layer:
        input_layer = args.input_layer
    if args.output_layer:
        output_layer = args.output_layer

    # Load the TensorFlow graph
    graph = load_graph(model_file)

    # Read and preprocess the input image for inference
    t = read_tensor_from_image_file(file_name,
                                    input_height=input_height,
                                    input_width=input_width,
                                    input_mean=input_mean,
                                    input_std=input_std)

    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)

    # Perform inference using the loaded graph
    with tf.compat.v1.Session(graph=graph) as sess:
        start = time.time()
        results = sess.run(output_operation.outputs[0], {input_operation.outputs[0]: t})
        end = time.time()

    # Squeeze the results and retrieve the top 5 predictions
    results = np.squeeze(results)
    top_k = results.argsort()[-5:][::-1]

    # Load labels and print the top predictions
    labels = load_labels(label_file)

    for i in top_k:
        return labels[i]
    
# Suppress TensorFlow informational messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set the size of the face detector window
size = 4

# Add the Graphviz binary path to the system environment variable
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/graphviz-2.38/release/bin/'

# Define the path to the input video and load the face cascade classifier
video_path = 'data/162_03_02.mp4'
classifier = cv2.CascadeClassifier('models/saved/haarcascade_frontalface_alt.xml')
global text

# Open a video capture object
webcam = cv2.VideoCapture(video_path) 

def faceBox(faceNet, frame):
    """
    Detect faces in a frame using a pre-trained deep learning model.

    Args:
    - faceNet: Face detection neural network model
    - frame: Input image frame

    Returns:
    - frame: Image frame with rectangles around detected faces
    - bboxs: List of bounding boxes for detected faces
    """
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], swapRB=False)
    faceNet.setInput(blob)
    detection = faceNet.forward()
    bboxs = []
    for i in range(detection.shape[2]):
        confidence = detection[0, 0, i, 2]
        if confidence > 0.7:
            x1 = int(detection[0, 0, i, 3] * frameWidth)
            y1 = int(detection[0, 0, i, 4] * frameHeight)
            x2 = int(detection[0, 0, i, 5] * frameWidth)
            y2 = int(detection[0, 0, i, 6] * frameHeight)
            bboxs.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
    return frame, bboxs

# Define file paths for face, age, and gender detection models
faceProto = "models/saved/opencv_face_detector.pbtxt"
faceModel = "models/saved/opencv_face_detector_uint8.pb"

# Load pre-trained models for face detection using OpenCV's dnn module
faceNet = cv2.dnn.readNet(faceModel, faceProto)

# Define constants for age and gender classification
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
padding = 20

# Get the current time
now = time.time() 
# Set a future time (e.g., 60 seconds from now)
future = now + 60

# Main loop
while True:
    # Read a frame from the webcam
    ret, frame = webcam.read()

    # Get face bounding boxes
    frame, bboxs = faceBox(faceNet, frame)

    # Process each face
    for bbox in bboxs:
        # Extract the face region with padding
        face = frame[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1),
                     max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]

        # Preprocess the face image
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)


        # Save the face image to a file and label it using an external module
        FaceFileName = "MERLB.jpg"
        cv2.imwrite(FaceFileName, face)
        text = main(FaceFileName)
        text = text.title()

        # Prepare the label for display
        label = "{}".format(text)

        # Display information based on detected emotion
        for emotion in ['Happy', 'Worried', 'Sad', 'Excited', 'Exhausted', 'Bored', 'Frustrated', 'Neutral']:
            if text == emotion:
                cv2.rectangle(frame, (bbox[0], bbox[1] - 30), (bbox[2], bbox[1]), (0, 255, 0), -1)
                cv2.putText(frame, label, (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                            cv2.LINE_AA)

    # Resize and display the frame
    frame = cv2.resize(frame, (950, 600))
    cv2.imshow('MERLB: Multimodal Emotion Recognition in Live Broadcasting', frame)

    # Wait for a key press and exit the loop if needed
    key = cv2.waitKey(30) & 0xff

# Check if the specified future time has passed
    if time.time() > future:
        try:
            # Close all OpenCV windows
            cv2.destroyAllWindows()

            # Determine the emotion, gender, and age categories, then select a random file based on the conditions
            if text == 'Happy':
                print('Happy')

            if text == 'Worried':
                print('Worried')

            if text == 'Sad':
                print('Sad')

            if text == 'Excited':
                print('Excited')

            if text == 'Exhausted':
                print('Exhausted')

            if text == 'Bored':
                print('Bored')

            if text == 'Frustrated':
                print('Frustrated')

            if text == 'Neutral':
                print('Neutral')
            # Break out of the loop after processing the emotion       
            break

        # Handle any exceptions and print an error message
        except :
            print('Please stay focus in Camera frame atleast 15 seconds & run this program again :)')
            break

    # Check if the 'Esc' key was pressed to exit the loop
    if key == 27:  
        break
