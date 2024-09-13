# MERLB: Multimodal Emotion Recognition in Live Broadcasting

<img src="https://camo.githubusercontent.com/2722992d519a722218f896d5f5231d49f337aaff4514e78bd59ac935334e916a/68747470733a2f2f692e696d6775722e636f6d2f77617856496d762e706e67" alt="Oryx Video-ChatGPT" data-canonical-src="https://i.imgur.com/waxVImv.png" style="max-width: 100%;">

# Overview

This paper presents a Multimodal Emotion Recognition in Live Broadcasting (MERLB) system that addresses challenges in real-time emotion analysis across modalities like body language, vocal tone, and facial expressions. The system uses deep convolutional neural networks with inception modules and dense blocks for facial recognition and tensor train layers for multimodal data fusion. 

# 👁️💬 Architecture

The proposed methodology comprises Deep CNN for image feature extraction, key segment selection for speech feature extraction, processing sequential data using Bi-LSTM, and conducting emotion classification using dense layers.

<img style="width: 80%; max-width: 100%;" src="/swerizwan/MERLB/resources/architecture.png" alt="MERLB Overview">

# MERLB Environment Setup

In our experiment, we utilized a TWITCH.TV dataset featuring ten English-speaking live broadcasters, ensuring gender balance. The dataset, comprising 7200 video clips, underwent annotation for emotional impact and live broadcasting behaviors. Class imbalance was addressed through an oversampling strategy. The input data for each clip included visual and audio frames. Our MERLB model, alongside other fusion models, underwent evaluation using F1-score.

The instructions for setting up a Conda environment named `merlb` with the required dependencies:

## Prerequisites

Ensure that you have [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) installed on your system.

## Setting Up the Environment

Follow these steps to create and activate the Conda environment with the specified packages:

1. **Create a Conda Environment**

   Open your terminal or command prompt and run the following command to create a Conda environment named `merlb`:

   ```bash
   conda create --name merlb python=3.6


This project requires Python 3.6 Compatibility with other versions is not confirmed. For a detailed list of requirements, refer to the `requirements.txt` file. Use the following command for easy installation:

```
pip install -r requirements.txt
```

# Dataset

## LoLs Dataset from Twitch.tv

- **Description:** The LoLs dataset comprises 5,513 training clips and 1,379 testing clips, totaling 9.56 hours.
- **Download:** [LoLs Dataset](https://drive.google.com/drive/folders/1IK5J6Yq701P0QzJjPANFXc8YmypO0TYe?usp=sharing)

## FIFA Dataset from Twitch.tv

- **Description:** The FIFA dataset includes 8,606 training clips and 2,151 testing clips, totaling 14.94 hours.
- **Download:** [FIFA Dataset](https://drive.google.com/drive/folders/1wSSoz6uMeCtuaVweNIMv7U7uJM6fUiQO?usp=sharing)

# Steps in Implementation

To run the experiments, follow these steps:

1. Run `utils/processingdataset.py` to convert the dataset into `.npy` files.
2. Optionally, utilize the oversampling technique outlined in the research paper by running `utils/annotatingaugment.py`.
3. After successfully completing these preprocessing tasks, run `training.py` to start the entire set of experiments.

# Live Broadcasting Demo

Once you have set up all the files and the dataset, you can run `live.py` to recognize emotions during live broadcasting.
