# Evolutionary Reinforcement Learning for OpenAI Gym
Implementation of Augmented-Random-Search for OpenAI Gym environments.
The paper can be found here: [Simple random search provides a competitive approach to reinforcement learning](https://arxiv.org/abs/1803.07055)

## Augmented-Random-Search (ARS)
ARS is an Evolutionary Strategy with a linear policy with weights w<sub>p.
Given an obervation s<sub>t<sub> an a ction a<sub>t is choosen by:

Continuous case:
a<sub>t<sub> = w<sub>p<sub> * s<sub>t<sub>

Discrete case:
a<sub>t<sub> = softmax(w<sub>p<sub> * s<sub>t<sub>)
  
The weights are mutated by adding i.i.d. normal distributed noise to every weight. 

w<sub>new<sub> = w<sub>p<sub> + &alpha; * N(0, 1)

Then the policy weights w<sub>p<sub> are updated in the direction of the best performing mutated weights.
