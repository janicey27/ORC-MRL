# Algorithm Introduction

This is the Github repo that contains the code for and the experiments run on the **Minimal Representaton Learning Algorithm**. 

Given a dataset with continuous state-action transitions across a number of steps, the model attempts to learn the minimal representation Markov Decision Process (MDP) that captures the dynamics of the environment solely by clustering datapoints based on **contradictions in transitions**. The trained model can then be used to predict the transitions of an unseen agent given its starting state, and can also be used to derive the optimal policy for a given agent to attain the highest reward.

For more information about the premise of the algorithm, please refer to the accompanying paper.

## Applying the Model
The model class named `MDP_model` can be imported from the `model.py` file found in the `Algorithm` folder. Once an instance of this model class is initialized, one can then run `model.fit()` or `model.fit_CV()` to train the model on the dataframe given, without or with cross validation. 

The format of the dataframe inputted needs to have the following columns: 
ID | TIME | FEATURE_1 | FEATURE_2 | ... | ACTION | RISK | OG_CLUSTER (Optional) |
---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | 
0 | 0 | ... | ... | ... | ... | ... | ... | 
0 | 1 | ... | ... | ... | ... | ... | ... | 
... | ... | ... | ... | ... | ... | ... | ... | 

The final `OG_CLUSTER` column is optional, used in the case that there is a "correct" clustering already known for the dataset, for purposes of calculating classification accuracy after training. 

For documentation on parameters used for fitting the model, **refer to the Jupyter Notebook `Maze/Maze_Experience`, where an example of model usage is outlined, with all the required parameters explained.**

# Repo Contents

1. **Algorithm** - this folder contains all the main code necessary to run the algorithm, including the model class `model.py`, clustering helper functions `clustering.py`, and testing helper functions `testing.py`. 
1. **Maze** - this folder contains all the code required to run the algorithm on solving a Maze reinforcement learning environment. 
    1. `Maze_Experience` is the first Jupyter notebook to open, which contains a very thorough explanation of the algorithm as applied to the problem.
    1. `Maze_Experiment_Replication` contains code that explains and replicates the large-scale experiments executed for the paper.
1. **Grid** - this folder contains a small experiment on that generates synthetic data to verify the model's performance on a 2D state space (similar to the maze, but without walls, and actions not translated physically to directions). `Grid_Experience` is the Jupyter Notebook with an explainer.
1. **COVID** - 

