# SAR data in Football/Soccer ⚽
[![Documentation Status](https://readthedocs.org/projects/openstarlab/badge/?version=latest)](https://openstarlab.readthedocs.io/en/latest/Pre_Processing/Sports/index.html)

## Introduction
This package offers functions to load and preprocess SAR data from various sources in football/soccer.

## Supported Data Providers
You can find detailed documentation on supported data providers [here](https://openstarlab.readthedocs.io/en/latest/Pre_Processing/Sports/SAR_data/Data_Provider/Soccer/index.html). The supported providers include:

- DataStadium
- Statsbomb with Skillcorner Tracking Data

## Supported Preprocessing Methods
For information on supported preprocessing methods, visit [this documentation](https://openstarlab.readthedocs.io/en/latest/Pre_Processing/Sports/SAR_data/Data_Format/Soccer/index.html). The available preprocessing methods are:

- State Action Reward (SAR) Format

## Examples
Here are some examples of how to download and preprocess data:

- **DataStadium Data:**
  - [Read the Docs Example](https://openstarlab.readthedocs.io/en/latest/Pre_Processing/Sports/SAR_data/Example/Soccer/Example_1/contents.html)
  - [Example Config File](https://github.com/open-starlab/PreProcessing/blob/master/example/config/datastadium/preprocessing_dssports2020.json)

- **StatsBomb and SkillCorner Data:**
  - [Read the Docs Example](https://openstarlab.readthedocs.io/en/latest/Pre_Processing/Sports/SAR_data/Example/Soccer/Example_2/contents.html)
  - [Example Config File](https://github.com/open-starlab/PreProcessing/blob/master/example/config/statsbomb_skillcorner/preprocessing_statsbomb_skillcorner2024.json)
    
## RL-ready datasets (DQN / QMIX)
If you are training RL models such as DQN (single-agent) and QMIX (multi-agent), you can convert the SAR `events.jsonl`
outputs into padded tensors with consistent action tokenization and train/val/test splits.

This produces a single shared multi-agent dataset with:
- `observation`: `(B, T, N, O)` (N=10 attackers)
- `action`: `(B, T, N)` (discrete action ids; default vocab size 16 with `PAD=15`)
- `reward`, `done`, `mask`: `(B, T)`
- `onball_mask`: `(B, T, N)` (for masking unavailable actions)

Notes:
- For DQN, you can flatten the agent dimension `N` into the batch dimension at load time.
- For QMIX, consume the tensors as-is.
