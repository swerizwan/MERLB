# Enhancing Emotion Recognition in Live Broadcasting: A Multimodal Deep Learning Framework

<img src="https://camo.githubusercontent.com/2722992d519a722218f896d5f5231d49f337aaff4514e78bd59ac935334e916a/68747470733a2f2f692e696d6775722e636f6d2f77617856496d762e706e67" alt="Oryx Video-ChatGPT" data-canonical-src="https://i.imgur.com/waxVImv.png" style="max-width: 100%;">

# Overview

This paper presents a Multimodal Emotion Recognition in Live Broadcasting (MERLB) system that addresses challenges in real-time emotion analysis across modalities like body language, vocal tone, and facial expressions. The system uses deep convolutional neural networks with inception modules and dense blocks for facial recognition and tensor train layers for multimodal data fusion. 

# 👁️💬 Architecture

The proposed methodology comprises Deep CNN for image feature extraction, key segment selection for speech feature extraction, processing sequential data using Bi-LSTM, and conducting emotion classification using dense layers.

<img style="max-width: 100%;" src="https://github.com/swerizwan/MERLB/blob/main/resources/architecture.png" alt="MERLB Overview">

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

# Experimental Results

The following table presents the F1 scores for different models (Early, Late, Joint, Deep, and Our method) across two datasets (FIFA and LoLs). The scores reflect performance in multi-task and single-task learning for various actions and emotions.

| Dataset        | Model  | Task   | Rou   | Pro   | Res   | Exp   | Fig   | Pun   | Def   | Dft   | VNeg  | VNeut | VPos  | ALow  | ANeut | AHigh |
|----------------|--------|--------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| **FIFA Dataset** | Early  | Multi | 0.535 | 0.563 | 0.586 | 0.396 | 0.653 | 0.512 | 0.786 | 0.723 | 0.794 | 0.389 | 0.623 | 0.213 | 0.796 | **0.475** |
|                |        | Single | 0.695 | 0.632 | 0.785 | 0.567 | 0.643 | **0.686** | 0.843 | 0.577 | 0.754 | 0.353 | 0.753 | 0.224 | 0.753 | 0.185 |
|                | Late   | Multi  | 0.753 | 0.696 | 0.735 | 0.593 | 0.696 | 0.586 | 0.864 | 0.686 | 0.797 | 0.543 | **0.863** | 0.243 | 0.754 | 0.253 |
|                |        | Single | 0.686 | 0.325 | 0.363 | 0.496 | 0.533 | 0.463 | 0.753 | 0.573 | 0.744 | 0.463 | 0.643 | 0.052 | 0.753 | 0.242 |
|                | Joint  | Multi  | 0.643 | **0.764** | 0.753 | 0.643 | 0.754 | 0.654 | 0.664 | 0.684 | 0.754 | 0.242 | 0.643 | 0.123 | 0.785 | 0.135 |
|                |        | Single | 0.613 | 0.643 | 0.744 | 0.632 | 0.612 | 0.533 | 0.675 | 0.721 | **0.823** | 0.354 | 0.753 | 0.213 | 0.812 | 0.124 |
|                | Deep   | Multi  | **0.754** | 0.753 | 0.743 | 0.713 | 0.567 | 0.674 | 0.754 | **0.784** | 0.785 | 0.634 | 0.842 | **0.282** | 0.853 | 0.321 |
|                |        | Single | 0.743 | 0.523 | 0.574 | **0.756** | 0.734 | 0.621 | 0.753 | 0.632 | 0.832 | 0.640 | 0.854 | 0.053 | 0.743 | 0.313 |
|                | Our    | Multi  | 0.696 | 0.743 | **0.854** | 0.753 | **0.765** | 0.621 | **0.857** | 0.742 | 0.869 | 0.538 | **0.859** | 0.206 | **0.880** | 0.254 |
|                |        | Single | 0.745 | 0.584 | 0.821 | 0.657 | 0.547 | 0.596 | **0.874** | 0.725 | 0.807 | **0.643** | 0.748 | 0.206 | 0.822 | 0.234 |
| **LoLs Dataset** | Early  | Multi | 0.686 | 0.564 | 0.564 | 0.456 | 0.563 | 0.524 | 0.861 | 0.764 | 0.821 | 0.486 | 0.689 | 0.153 | 0.869 | **0.354** |
|                |        | Single | 0.753 | 0.643 | 0.834 | 0.643 | 0.597 | **0.765** | 0.855 | 0.654 | 0.846 | 0.467 | 0.816 | 0.133 | 0.814 | 0.269 |
|                | Late   | Multi  | **0.884** | 0.754 | 0.832 | 0.645 | 0.754 | 0.643 | 0.879 | 0.564 | 0.842 | 0.521 | **0.873** | 0.121 | 0.883 | 0.311 |
|                |        | Single | 0.686 | 0.435 | 0.453 | 0.654 | 0.645 | 0.564 | 0.862 | 0.654 | 0.866 | 0.563 | 0.754 | 0.075 | 0.835 | 0.321 |
|                | Joint  | Multi  | 0.723 | 0.578 | 0.671 | 0.543 | 0.676 | 0.672 | 0.767 | 0.798 | 0.875 | 0.367 | 0.764 | 0.203 | 0.853 | 0.321 |
|                |        | Single | 0.645 | 0.753 | **0.877** | 0.785 | 0.631 | 0.653 | 0.795 | 0.785 | 0.851 | 0.474 | 0.821 | 0.212 | 0.821 | 0.264 |
|                | Deep   | Multi  | 0.874 | **0.784** | 0.786 | 0.732 | 0.675 | 0.721 | **0.897** | 0.676 | **0.887** | 0.642 | 0.854 | **0.231** | 0.875 | 0.342 |
|                |        | Single | 0.721 | 0.542 | 0.521 | 0.735 | 0.734 | 0.621 | 0.754 | 0.653 | 0.845 | 0.563 | 0.762 | 0.153 | 0.825 | 0.276 |
|                | Our    | Multi  | **0.895** | 0.753 | **0.876** | **0.865** | 0.845 | 0.742 | 0.879 | 0.812 | 0.902 | **0.643** | 0.875 | 0.325 | **0.892** | 0.412 |
|                |        | Single | 0.786 | 0.623 | 0.821 | 0.765 | **0.854** | 0.685 | 0.879 | **0.845** | **0.892** | 0.632 | 0.863 | 0.253 | 0.854 | **0.412** |

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
