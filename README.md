# Algorithm Introduction

This is the Github repo that contains the code for and the experiments run on the **Minimal Representation Learning Algorithm (MRL)**. 

Given a dataset with continuous state-action transitions across a number of steps, the model attempts to learn the minimal representation Markov Decision Process (MDP) that captures the dynamics of the environment solely by clustering datapoints based on **incoherences in transitions** (also refered to as contradictions). The trained model can then be used to predict values of the system given a starting state and a decision policy or sequence of actions, and can also be used to derive the optimal policy for a given agent to attain the highest reward.

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
1. **Maze** - this folder contains all the code required to run the algorithm on an example: the Maze RL environement (Sutton and Barto 2018). This experiment as well as a discussion of its results are presented in the paper.
    1. `Maze_Experience` is the first Jupyter notebook to open, which contains a very thorough explanation of the algorithm as applied to the problem.
    1. `Maze_Experiment_Replication` contains code that explains and replicates the large-scale experiments executed for the paper.
1. **Grid** - this folder contains a mini experiment on that generates synthetic data to verify the model's performance on a 2D state space (similar to the maze, but without walls, and actions not translated physically to directions). `Grid_Experience` is the Jupyter Notebook with an explainer.
1. **COVID** - this folder contains the preliminary version of the code updated to fit the model to COVID-19 Prediction. The `Covid_Explainer` notebook contains the code and explanations for how the model was adapted to predict COVID-19 case trends.
1. **HIV** - this folder contains code used to run experiments on HIV indicators. 
1. **Toy** - this folder contains a mini experiment where the model is trained on a state space of concentric circles of expanding radii. 
