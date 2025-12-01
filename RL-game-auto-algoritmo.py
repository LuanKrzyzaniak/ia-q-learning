import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import pickle
from config import STEP_REWARD, WALL_REWARD, HOLE_REWARD, GOAL_REWARD
from config import ALPHA, GAMMA, EPSILON, SIMMULATION_NUMBER, ALPHA_DECAY, EPSILON_DECAY, DECAY_STEP, TRAIN, RENDERS
import pandas as pd

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed

# Parametros
dqn_param_grid = [
    {"learning_rate": 0.001, "buffer_size": 25000, "batch_size": 32, "gamma": 0.99},
    {"learning_rate": 0.0007, "buffer_size": 50000, "batch_size": 32, "gamma": 0.99},
    {"learning_rate": 0.0005, "buffer_size": 25000, "batch_size": 32, "gamma": 0.98},
    {"learning_rate": 0.0003, "buffer_size": 50000, "batch_size": 32, "gamma": 0.98},
    {"learning_rate": 0.0001, "buffer_size": 25000, "batch_size": 32, "gamma": 0.97},
    {"learning_rate": 0.001, "buffer_size": 50000, "batch_size": 64, "gamma": 0.97},
    {"learning_rate": 0.0007, "buffer_size": 25000, "batch_size": 64, "gamma": 0.95},
    {"learning_rate": 0.0005, "buffer_size": 50000, "batch_size": 64, "gamma": 0.95},
    {"learning_rate": 0.0003, "buffer_size": 25000, "batch_size": 64, "gamma": 0.90},
    {"learning_rate": 0.0001, "buffer_size": 50000, "batch_size": 64, "gamma": 0.90}
]

results = []

class MazeEnv(gym.Env):
    def __init__(self):
        super(MazeEnv, self).__init__()

        # Define grid size
        self.height = 15
        self.width = 15

        self.turn = 0

        # Define elements
        self.EMPTY = 0
        self.WALL = 1
        self.HOLE = 2
        self.AGENT = 3
        self.GOAL = 4
        self.agent_pos = None
        self.goal_pos = None
        self.goal_id = None
        
        # Initialize visualization
        self.fig, self.ax = plt.subplots(figsize=(7, 5))
        plt.ion()

        # Action space: up, right, down, left
        self.action_space = spaces.Discrete(4)
        # Observation space: single number representing state
        self.observation_space = spaces.MultiDiscrete([5, 5, 5, 5, 4, 15*15])

        # Create grid
        self.grid = None
        self.grid_initialization()

    def grid_initialization(self):
        self.turn = 0
        
        # Create grid
        self.grid = np.zeros((self.height, self.width))
        
        # Set agent, enemy and goal position
        self.agent_pos = [random.randint(2, self.height - 3), random.randint(2, self.width - 3)]

        self.goal_id = random.randint(1,4)
        if self.goal_id == 1:
            self.goal_pos = [1, 1]
        elif self.goal_id == 2:
            self.goal_pos = [1, self.width - 2]
        elif self.goal_id == 3:
            self.goal_pos = [self.height - 2,1]
        else:
            self.goal_pos = [self.height - 2, self.width - 2]
        
        # Add border walls
        self.grid[0:self.height, 0] = self.WALL
        self.grid[0, 0:self.width ] = self.WALL
        self.grid[self.height-1, 0:self.width] = self.WALL
        self.grid[0:self.height , self.width-1] = self.WALL
        # Set initial positions
        self.grid[self.agent_pos[0], self.agent_pos[1]] = self.AGENT
        self.grid[self.goal_pos[0], self.goal_pos[1]] = self.GOAL
        # Add inner walls
        for i in range(1, self.height):
            for j in range(1, self.width):
                if self.grid[i][j] == self.EMPTY:
                    r = random.random() 
                    if r < 0.2:
                        self.grid[i][j] = self.WALL
                    elif r < 0.25:
                        self.grid[i][j] = self.HOLE       

    def get_state(self, isnumpy=True):
        up_view = self.grid[self.agent_pos[0] - 1][self.agent_pos[1]]
        down_view = self.grid[self.agent_pos[0] + 1][self.agent_pos[1]]
        left_view = self.grid[self.agent_pos[0]][self.agent_pos[1] - 1]
        right_view = self.grid[self.agent_pos[0]][self.agent_pos[1] + 1]
        agent_position = self.agent_pos[0] * self.width + self.agent_pos[1]
        observation = (up_view, down_view, left_view, right_view, self.goal_id - 1, agent_position)
        if isnumpy:
            return np.array(observation)
        else:
            return observation

    def step(self, action, isnumpy = True):
        # Add 1 turn
        self.turn += 1

        # Save previous position
        prev_pos = self.agent_pos.copy()

        # Move agent
        if action == 0:  # up
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 1:  # right
            self.agent_pos[1] = min(self.width - 1, self.agent_pos[1] + 1)
        elif action == 2:  # down
            self.agent_pos[0] = min(self.height - 1, self.agent_pos[0] + 1)
        elif action == 3:  # left
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)

        # Check if new position is valid
        new_pos_value = self.grid[self.agent_pos[0], self.agent_pos[1]]

        # Define rewards and terminal states
        done = False
        reward = STEP_REWARD()  # small negative reward for each step

        if new_pos_value == self.WALL:
            self.agent_pos = prev_pos  # revert move
            reward = WALL_REWARD()
        elif new_pos_value == self.HOLE:
            done = True
            reward = HOLE_REWARD()
        elif self.agent_pos == self.goal_pos:
            done = True
            reward = GOAL_REWARD()

        # Update grid
        if new_pos_value != self.WALL:
            self.grid[prev_pos[0], prev_pos[1]] = self.EMPTY
            self.grid[self.agent_pos[0], self.agent_pos[1]] = self.AGENT

        if self.turn == 200:
            return self.get_state(isnumpy), reward, True, False, {}

        return self.get_state(isnumpy), reward, done, False, {}

    def reset(self, *, seed = None, options = None, return_info = False, isnumpy = True):
        super().reset(seed=seed)
        # Reset game to initial state
        self.turn = 0
        self.grid_initialization()
        return self.get_state(isnumpy), {}

    def render(self):
        self.ax.clear()
        # Define colors for each element
        colors = {
            self.EMPTY: 'white',
            self.WALL: 'gray',
            self.HOLE: 'black',
            self.AGENT: 'blue',
            self.GOAL: 'green',
        }

        name = {
            self.EMPTY: 'Vazio',
            self.WALL: 'Parede',
            self.HOLE: 'Buraco',
            self.AGENT: 'Agente',
            self.GOAL: 'Objetivo',
        }

        # Create color map
        cmap = plt.cm.colors.ListedColormap(list(colors.values()))
        # Plot the grid
        self.ax.imshow(self.grid, cmap=cmap)

        # Add legend
        legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=color, label=name[key])
                           for key, color in colors.items()]
        self.ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))

        plt.axis('off')
        plt.pause(0.1)
        self.fig.canvas.draw()


class QLearningAgent:

    def __init__(self, action_space, learning_rate, discount_factor, epsilon):
        self.q_table = defaultdict(lambda: np.zeros(action_space.n))
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.action_space = action_space

    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(dict(self.q_table), f)

    def load_model(self, filename):
        with open(filename, 'rb') as f:
            self.q_table = defaultdict(lambda: np.zeros(self.action_space.n), pickle.load(f))

    def get_action(self, state):
        if random.random() < self.epsilon:
            return self.action_space.sample()

        q_values = self.q_table[state]
        exp_q = np.exp(q_values - np.max(q_values))
        probs = exp_q / np.sum(exp_q)
        return np.random.choice(len(q_values), p=probs)

    def update(self, state, action, reward, next_state):
        old_value = self.q_table[state][action]
        next_max = np.max(self.q_table[next_state])
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_table[state][action] = new_value

# Training Model/Agent
if TRAIN():
    # Consts
    summary_results = []

    for idx, params in enumerate(dqn_param_grid, 1):
        print(f"\nTreinando modelo DQN nº {idx} com params: {params}")

        env = DummyVecEnv([lambda: MazeEnv()])

        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=params["learning_rate"],
            buffer_size=params["buffer_size"],
            batch_size=params["batch_size"],
            gamma=params["gamma"],
            verbose=0
        )

        model.learn(total_timesteps=SIMMULATION_NUMBER())

        # Save model
        model_path = f"../output/models/sb3_dqn_model_{idx}"
        model.save(model_path)

# Test the trained agent
else:
    # Consts
    summary_results = []
    episodes = 1000
    
    for idx in range(1, 11):
        print(f"\nTestando modelo nº{idx}")
        
        # Métricas
        total_reward = 0
        success = 0
        
        env = MazeEnv()
        state, _ = env.reset(isnumpy = False)
        done = False
        
        if RENDERS():
            env.render()
            
        # Load the trained agent
        model_path = f"../output/models/sb3_dqn_model_{idx}"
        model = DQN.load(model_path)


        for episode in range(episodes):
            state, _ = env.reset(isnumpy=False)
            done = False

            while not done:
                action, _ = model.predict(state, deterministic=True)
                state, reward, done, _, _ = env.step(action, isnumpy=False)
                total_reward += reward
                
                #print(f"Action: {action}, Reward: {reward}")  
                if RENDERS():
                    env.render()
                    
                if reward == GOAL_REWARD():
                    success += 1
                    
                
        mean_reward = total_reward / episodes
        success_rate = success / episodes

        print(f"Modelo {idx}: Recompensa média = {mean_reward:.2f}, Taxa de sucesso = {success_rate:.2f}")

        summary_results.append({
            "Modelo": idx,
            "Mean Reward": mean_reward,
            "Success Rate": success_rate
        })

    df = pd.DataFrame(summary_results)
    df.to_csv('../output/test_summary_results_sb3_dqn.csv', index=False)
    print("Resultados de teste salvos em '../output/test_summary_results_sb3_dqn.csv'")