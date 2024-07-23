# Event Data in Football/Soccer ⚽
[![Documentation Status](https://readthedocs.org/projects/openstarlab/badge/?version=latest)](https://openstarlab.readthedocs.io/en/latest/Pre_Processing/Sports/index.html)
## Introduction
This package offers functions to load and preprocess event data from various sources in football/soccer.

## Supported Data Providers
You can find detailed documentation on supported data providers [here](https://openstarlab.readthedocs.io/en/latest/Pre_Processing/Sports/Event_data/Data_Provider/index.html). The supported providers include:

- DataFactory
- Metrica
- Opta
- Robocup 2D Simulation
- Sportec
- Statsbomb
- Statsbomb with Skillcorner Tracking Data
- Wyscout

## Supported Preprocessing Methods
For information on supported preprocessing methods, visit [this documentation](https://openstarlab.readthedocs.io/en/latest/Pre_Processing/Sports/Event_data/Data_Format/index.html). The available preprocessing methods are:

- Unified and Integrated Event Data (UEID)
- NMSTPP (same format required for [Football Match Event Forecast](https://github.com/calvinyeungck/Football-Match-Event-Forecast))
- Other Event Data Formats

## Examples
Here are some examples of how to download and preprocess data:

- **Wyscout Data (NMSTPP format):**
  - [Read the Docs Example](https://openstarlab.readthedocs.io/en/latest/Pre_Processing/Sports/Event_data/Example/Football/Example_1/contents.html)
  - [GitHub Example](https://github.com/open-starlab/PreProcessing/blob/master/example/NMSTPP_data.py)

- **StatsBomb and SkillCorner Data:**
  - [Read the Docs Example](https://openstarlab.readthedocs.io/en/latest/Pre_Processing/Sports/Event_data/Example/Football/Example_2/contents.html)
  - [GitHub Example](https://github.com/open-starlab/PreProcessing/blob/master/example/statsbomb_skillcorner.py)
