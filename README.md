# carrom-with-wfnn

CS 747 ASSIGNMENT 4

CARROM BOARD SIMULATION USING REINFORCEMENT LEARNING


The agent uses a WIRE FITTED NEURAL NETWORK (WFNN) model for predicting the values of actions.

This WFNN training model has a neural network and a wire fitting interpolator.

It basically works in 3 steps:

1. Feed the state into the neural network. From the output of the neural network, find the action with the highest q. Execute the action. Record an experience composed of initial state, action, next state and the reward received as a result of the action.


2. Computation of a new estimate of Q values using the action with the maximum q values (one step Q-learning equation)


3. Fitting the wires with an interpolated curve and calculation of wire fitter partial derivatives to calculate desired action and q values. Lastly, train the neural network using backpropagation.


An optimum weight is achieved which clears the board in 26 turns. 

I have provided a script.sh that directly runs the simulation. (FOr the saved value of the optimum weight. No training required.)

For viewing the backpropagation part, one can use the commented lines in the script.sh code.


References:

https://openresearch-repository.anu.edu.au/bitstream/1885/47080/6/02whole.pdf
 
Carrom board simulation taken from:
https://github.com/samiranrl/Carrom_rl

TEAM NAME: INNOVATORS
Team members:
MEHTA NIHAR NIKHIL
VENKAT KALYAN 
