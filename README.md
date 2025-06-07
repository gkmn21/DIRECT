# DIRECT: Deep Reinforcement Learning for Tourist Route Recommendation

Source code for the paper **DIRECT: Deep Reinforcement Learning for Tourist Route Recommendation**.


## Dependencies
The `requirements.txt` has been provided for creating a virtual environment with all dependencies. To install the requirements run:
```
pip install -r requirements.txt
```

Dependencies include:
- Python 3.12.4
- PyTorch 2.7.0
- `stable-baselines3`
- `gymnasium`

## Dataset Generation
The folder `dataset_generation` contains scripts to generate datasets.

- Run script `dataset_generation/prepare_data.py` to fetch data from OSM for a region and prepare train, validation and test datasets. The generated dataset files will be saved in `data` folder.

## DIRECT Model
- Run script `run_model.py` to train the model.
- Run script `inference.py` to run inference on the test set.
- Finally, run script `eval.py` to obtain the evaluation metrics.

## Baselines
- The scripts for naive baselines SP and GP is present in the folder `Baselines`.
- Source code provided by the publications was used for baselines RB&C and D-RL


