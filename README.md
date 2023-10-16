# Effect of Euclidean vs Projective geometry on curiosity based exploration

This project is an implementation exhibiting the effect of Euclidean vs Projective geometry on curiosity based exploration.
The setup of each simulation is an agent and an object, both represented as points in the 2D euclidean space. The agent holds beliefs about the position of the object in its internal space, which is either Euclidean or Projective. The agent selects a translation at each step based on the information gained (epistemic value) through each action.


## Requirements

This project requires python 3. The required dependencies can be installed with `pip install -r requirements.txt`.

## Running the code

Simulations can be produced by executing the main file :

`python main.py`

This will write .json files containing the results of the simulations in a directory called "sims".

The results of the simulation can be visualized with several commands :

- `python view_trajectories.py` will show the trajectory of the agent for each simulation.

- `python view_losses.py` will plot the average losses per actions for all simulations. 

- `python view_losse_ranges.py` will display the min, max and average loss range (difference between best and worst action) per simulation.


