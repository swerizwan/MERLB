# MERLB: Multimodal Emotion Recognition in Live Broadcasting
In our experiment, we utilized a TWITCH.TV dataset featuring ten English-speaking live broadcasters, ensuring gender balance. The dataset, comprising 7200 video clips, underwent annotation for emotional impact and live broadcasting behaviors. Class imbalance was addressed through an oversampling strategy. The input data for each clip included visual and audio frames. Our MERLB model, alongside other fusion models, underwent evaluation using F1-score.

# Experiment Requirements
This project requires the use of Python 3.6.3; however, there is no confirmation of its compatibility with other versions. For a detailed list of requirements, see the requirements.txt file. The following command can be used to install these easily: 
pip install -r requirements.txt.

# Dataset

LoLs dataset from twitch.tv
The dataset along with csv files for training 5,513 clips and testing of 1,379 clips, total duration is 9.56 hours can be downloaded by the below link: https://drive.google.com/drive/folders/1IK5J6Yq701P0QzJjPANFXc8YmypO0TYe?usp=sharing

FIFA dataset from twitch.tv
The dataset along with csv files for training 8,606 clips and testing of 2,151 clips, total duration is 14.94 hours can be downloaded by the below link: https://drive.google.com/drive/folders/1wSSoz6uMeCtuaVweNIMv7U7uJM6fUiQO?usp=sharing

# Steps in Implementation
Follow the below steps to run the experiments:
•	Run utils/processingdataset.py to start the process of converting the dataset into npy files.
•	Utilize the oversampling technique outlined in the research paper by running utils/annotatingaugment.py as an optional step.
•	After completing these preprocessing tasks successfully, run training.py to start the entire set of experiments.

# Live Broadcasting Demo
After setting up all the files and the dataset, you can run 'live.py' to recognise emotions during live broadcasting.
