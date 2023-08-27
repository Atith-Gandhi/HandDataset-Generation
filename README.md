# Codebase for generating hand images with additional channels

## Overview
- input: one hand image (HWC - 3 channels: RGB)
- 1st additional channel: increasing color values from the thumb to little finger.
- 2nd additional channel: increasing color values top of fingers to the base of hand.
- 3nd additional channel: different color denotes dorsal/ventral.

Note1: If you wanna review the codebase, you could just focus on codes with comments start with `yy: ` 

Note2: I have included the folders `generated_mh_info` and `dataset_additional_info` as example. You can find how to generate these two folders in the code `generate_mh_info.py` and `generate_dataset.py`.

## Installation
1. Clone this repo
2. Follow the [minimal hand](https://github.com/CalciferZh/minimal-hand) repo's requirement to install the required packages.
3. Download the [CMU dataset](http://domedb.perception.cs.cmu.edu/panopticDB/hands/hand_labels_synth.zip). (You could download my [local dataset](https://drive.google.com/file/d/1cAJazNbQB2_bVIivb-gm-UDdKDaAXl-d/view?usp=share_link) (i.e., `CMU_dataset_5000_ori` & `CMU_keypoints_5000`) directly via google drive as well)

## Folder structure
```
.
├── datasets
│   ├── CMU_dataset_5000_ori
│   └── CMU_keypoints_5000
└── minimal-hand
```

## Usage
1. Generate minimal hand: `python generate_mh_info.py`
2. Generate the dataset: `python generate_dataset.py`
