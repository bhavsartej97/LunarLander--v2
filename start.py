## Import Dependancies
import gym
import random
import math
import numpy as np
import os
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

## Set Parameters
env = gym.make('LunarLander-v2')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
# batch for gradient descent. Will be implementing mini-batch gradient descent.
batch_size = 32
# No of games we want out agent to play
no_episodes = 1000
# Output directory to store output
output_dir = 'model_output/lunarlander'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

## Constants
inf=99999

## Define Agent
class DQNAgent:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        ## For each episode we play, we store a random value, rewards and
        ## use this data to train the model. Random because more diversity
        self.memory = deque(maxlen=2000)
        ## discount factor
        self.gamma = 0.9
        ## Initial Exploration Factor
        self.episolon = 1.0  ## Skewed towards exploration
        ## Episolon decay
        self.epislon_decay = 0.995
        ## Final Exploration Factor
        self.min_epsilon = 0.01
        ## Gradient Descent Learning Rate
        self.learning_rate = 0.001

        self.model = self.build_model()

    ## Defines a Dense Neural Network for Approximating Q*
    def build_model(self):
        ## Setting up a Keras Model
        model = Sequential()

        ## Defining a dense layer in the NN, with 24 neurons.
        ## In the context of artificial neural networks, the rectifier is an activation function,
        ## defined as the positive part of its argument,
        ## f(x) = x* = max(0, x)
        # where x is the input to a neuron.
        model.add(Dense(24, input_dim = state_size, activation = 'relu'))
        model.add(Dense(24, activation='relu'))

        ## Defining output Layers
        ## No of output layers = action_size, 1 for each Action.
        model.add(Dense(self.action_size, activation='linear'))

        ## Complie the model
        model.compile(loss = 'mse', optimizer=Adam(lr=self.learning_rate))

        return model

    ## Given a state-action pair, what next state do we end up in and what reward do we get
    ## for the state-action pair.
    def remember(self, state, action, reward, nexttState, done):
        self.memory.append((state, action, reward, nexttState, done))

    ## We figure out what action to take, given the state.
    ## We either explore randomly or exploit provided we have a good enough policy
    def act(self, state):
        # Exploration
        if np.random.rand() <= self.min_epsilon:
            ## return a random action.
            return random.randrange(self.action_size)
        # Exploitation
        # WE use our deep learning model to guess/predict what the best possible action is.
        act_values =self.model.predict(state)
        return np.argmax(act_values[0])

    ##
    def replay(self, batch_size):
        # gets a random sample form our game history
        mini_batch = random.sample(self.memory, batch_size)

        for state, action, reward, nexttState, done in mini_batch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(nexttState)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target

            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.episolon > self.min_epsilon:
            self.episolon *= self.epislon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


## Helper Functions
def getDist(goalx, goaly, llx, lly):
    return math.sqrt(math.pow(goalx - llx, 2) + math.pow(goaly - lly, 2))


## Init Agent
agent = DQNAgent(state_size, action_size)
## Interact with the env
done = False
for e in range(no_episodes):
    ## Starts the env.
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    ## Iterate over max time steps i.e. max game time
    for time in range(1000):
        #env.render()
        ## get initial action
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        if not done:
            if reward > 0:
                ## calculate the distance between centere(0, 0) and the spot where the
                ## lander landed.
                xpos, ypos, _, _, _, _, _, _ = next_state
                dist = getDist(0,0, xpos, ypos)
                ## FIXME
                reward+=inf
                print("Rewards = " + str(reward))
        else:
            reward = reward
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        if done:
            print("Episode : {}/{}, Score = {}, e : {:.2}".format(e, no_episodes, reward, agent.episolon))
    ## Train theta weights
    if len(agent.memory) > batch_size:
        agent.replay(batch_size)