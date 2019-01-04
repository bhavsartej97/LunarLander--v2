from ANN import ANN
from collections import deque
from keras.optimizers import Adam
import csv
import gym
import numpy as np
import random

DATA_DIR = 'data/'

## This class represents a RL Agent that will play this game.
class Agent:
    def __init__(
            self, name, env,

            # Q-learning hyperparameters
            epsilon_decay=0.998,
            epsilon_max=1.0,
            epsilon_min=0.05,
            gamma=0.99,

            # ANN hyperparameters
            alpha=0.0001,
            layers=None,
            activation='relu',
            loss='mean_squared_error',

            # Experience Replay hyperparameters
            memory_max=2 ** 16,
            memory_min=2 ** 6,
            batch_size=2 ** 6,
    ):
        if layers is None:
            layers = [24, 24]
        self.name = name
        self.env = env

        ## This creates a deque of memory_max size.
        self.memory = self.create_memory(memory_max)
        self.memory_min = memory_min
        self.memory_max = memory_max

        ## No of states
        self.ns = env.observation_space.shape[0]
        ## Number of actions
        self.na = env.action_space.n
        print("Input Neurons = " + str(self.ns) + " Output Neurons = " + str(self.na))
        ## Mini-batch size.
        self.batch_size = batch_size

        ## Exploration Factor
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_max
        self.epsilon = self.epsilon_max
        ## Discount Factor
        self.gamma = gamma
        ## learning Rate
        self.alpha = alpha
        ## Sze of hidden layers
        self.layers = layers

        ## The agents states learning at a particular point of time, i.e. when we have enough data for the current
        ## mini-batch size.
        self.is_learning = False

        ## Creates the neural network.
        self.ann = self.create_ann('{}.ann'.format(name), alpha, layers, activation, loss)
        ## Creates the target network, we need the target network to train the og. neural network based on the error.
        self.target_ann = self.create_ann('{}.target_ann'.format(name), alpha, layers, activation, loss)

        ## Number of steps taken by the agent in one test simulation.
        self.frames = 0

    # ANN
    ## This is described better in the ANN.py file.
    def create_ann(self, name, alpha, layers, activation, loss):
        optimizer = Adam(lr=alpha)

        ## Create an ANN instance.
        ann = ANN(
            name,
            input_dim=self.ns,
            output_dim=self.na,
            activation=activation,
            layers=layers,
            loss=loss,
            optimizer=optimizer
        )

        return ann

    # Experience Replay
    ## A buffer to store the past experiences.
    def create_memory(self, memory_max):
        return deque([], maxlen=memory_max)

    ## Store a sample in memory
    def store_experience(self, experience):
        self.memory.append(experience)


    def get_batch(self):
        # Sample
        ## Get a set of random samples form experience buffer of the batch size.
        E = random.sample(self.memory, self.batch_size)
        ## S = all States.
        S = np.array([e['s'] for e in E])
        ## All Alctions
        A = np.array([e['a'] for e in E])
        ## All Rewards
        R = np.array([e['r'] for e in E])
        ## All S', i,.e T(S[i], A[i]) -> S'[i], R[i]
        T = np.array([e["s'"] for e in E])
        ## Is Done
        D = np.array([e['done'] for e in E])

        # Compute
        ## Predict the q-values for each state
        q = self.ann.predict(S)
        ## Predict target q-values for each S'.
        q_t = self.target_ann.predict(T)
        ## init target value buffer
        y = np.zeros((len(E), self.na))
        for i in range(len(E)):
            ## for each reward, action and the predicted q-values for state i.
            r, a, t = R[i], A[i], q[i]
            ## No further rewards if done.
            if D[i]:  # if done
                t[a] = r
            ## r[i] + gamma * max_a Q(S', a; theta)
            else:
                t[a] = r + self.gamma * np.max(q_t[i])
            ## set the target predicted value
            y[i] = t

        return S, y

    # Agent interface
    ## This will return true if there are enough samples available in experience buffer
    ## to actually start learning.
    def can_learn(self):
        if self.is_learning:
            return True
        elif len(self.memory) >= self.memory_min:
            print('[INFO] Started learning!')
            self.is_learning = True
            return True
        else:
            return False

    ## Returns either a random action with probability epsilon or the best predicted action
    ## based on our current neural network.
    def get_action(self, state):
        # Exploration
        # Epsilon greedy action selection
        # Select random action with probability epsilon.
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        # Exploitation.
        # Return the best action using our neural network.
        else:
            reshaped = np.asarray(state).reshape((1, self.ns))
            return np.argmax(self.ann.predict([reshaped]))

    ## Remember the state, action and the received reward as well as the next state.
    def experience(self, s, a, r, s_, done):
        experience = {
            's': s, 'a': a, 'r': r, "s'": s_, 'done': done
        }
        self.store_experience(experience)
        self.frames += 1
        return experience

    ## Start learning
    def learn(self):
        if self.can_learn():
            ## Get a batch from memory
            X, y = self.get_batch()
            ## Train using that batch.
            self.ann.train(X, y, batch_size=self.batch_size)
            return True
        else:
            return False

    ## Update Weights and Parameters
    def update(self, ann=True):
        if ann:
            # Update weights of target network.
            self.target_ann.set(self.ann.get())
        # Update Epsilon
        self.epsilon *= self.epsilon_decay

    ## Reset the number of steps after a sample run.
    def clear_frames(self):
        self.frames = 0


## This class represents a Trial run using the algorithm defined above.
class Trial:
    def __init__(
            self,
            name='default',
            episodes=2000,
    ):
        ## Define the environment
        env = gym.make('LunarLander-v2')

        self.episodes = episodes
        self.name = name

        ## This stores all the rewards at the end of each episode.
        self.rewards = []
        ## This is used for calculating the average over 100 episodes.
        ## This will be used while comparing the performance of our algorithm w.r.t the leader-board.
        self.running_rewards = deque([], maxlen=100)

        ## Create an agent.
        self.agent = Agent(name, env)
        self.env = env

        ## We are storing a map of the episodes and the score in that episode for some data visualization and
        ## comparing the effect of different values for hyper-parameters
        self.score_file = open('data/score_reports' + str(self.agent.batch_size) + str(self.agent.layers[0])
                               + str(self.agent.layers[1]) + '.csv', mode='w')
        self.score_file.write("Reports for LunarLander with HyperParameters:")
        self.score_file.write("alpha = " + str(self.agent.alpha) + " Batch Size = " + str(self.agent.batch_size) +
                              " Gamma = " + str(self.agent.gamma) + " H1 = " + str(self.agent.layers[0]) +
                              " H2 = " + str(self.agent.layers[1]) + "\n")
        self.writer = csv.writer(self.score_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    ## This simulates one run of DQN Algorithm.
    def run(self):

        agent = self.agent
        env = self.env

        for episode in range(self.episodes):
            agent.clear_frames()
            total_reward = 0
            ## get the first state.
            state = env.reset()
            done = False
            while not done:
                ## avoid this for speed
                env.render()
                ## get action based on state.
                action = agent.get_action(state)
                ## take the action and get next state, reward, etc.
                next_state, reward, done, _unused = env.step(action)
                ## store the experience.
                agent.experience(state, action, reward, next_state, done)
                ## learn based on the data gathered.
                agent.learn()
                ## update reward
                total_reward += reward
                ## update state / move forward.
                state = next_state

            ## update the weights the simulation of 1 episode.
            agent.update(ann=True)
            ## save rewards.
            self.rewards.append(total_reward)
            self.running_rewards.append(total_reward)
            ## calculate mean for record keeping.
            mean_reward = self.mean_reward()

            ## wirte data to a CSV file.
            self.writer.writerow([episode, mean_reward])
            print('#{}, Reward: {:.2f}, Epsilon: {:.2f}, Frames: {}; Running: {:.2f}'.format(episode,
                                                                                             total_reward,
                                                                                             agent.epsilon,
                                                                                             agent.frames,
                                                                                             mean_reward))
            ## This is our goal. Since OpenAI defines LunarLander to be solved if we gen an average of >=200.00 over
            ## 100 consecutive episodes.
            if mean_reward > 200:
                if len(self.rewards) > 100:
                    print("[INFO] We are doing good")
            elif mean_reward < -100 and episode >= 1000:
                print("[INFO] This algorithm is not working out. Need some change")
                break

        print('* Completed "{}" in {} episodes with {:.2f} mean reward'.format(self.name, episode, mean_reward))


    def mean_reward(self):
        return np.mean(self.running_rewards)


## main Method
if __name__ == '__main__':
    trial = Trial('default')
    trial.run()
