# MERLB: Multimodal Emotion Recognition in Live Broadcasting
In our experiment, we utilized a TWITCH.TV dataset featuring ten English-speaking live broadcasters, ensuring gender balance. The dataset, comprising 7200 video clips, underwent annotation for emotional impact and live broadcasting behaviors. Class imbalance was addressed through an oversampling strategy. The input data for each clip included visual and audio frames. Our MERLB model, alongside other fusion models, underwent evaluation using F1-score.

# Experiment Requirements

This project requires Python 3.6.3. Compatibility with other versions is not confirmed. For a detailed list of requirements, refer to the `requirements.txt` file. Use the following command for easy installation:

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
