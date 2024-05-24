################################### Imports ###################################

import os
import gym
import csv
import time
import glob
import random
import logging
import argparse
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from time import sleep
from gym.spaces import Box
from scipy.stats import norm
from matplotlib import pyplot as plt

# expert_prior.py
from tensorflow.keras.layers import Dense, Conv2D, GlobalAveragePooling2D, Concatenate
from tf2rl.algos.policy_base import OffPolicyAgent
from tf2rl.misc.target_update_ops import update_target_variables
from tf2rl.policies.tfp_gaussian_actor import ExpertGuidedGaussianActor

# tf2rl trainer.py
from tf2rl.experiments.utils import save_path, frames_to_gif
from tf2rl.misc.get_replay_buffer import get_replay_buffer
from tf2rl.misc.prepare_output_dir import prepare_output_dir
from tf2rl.misc.initialize_logger import initialize_logger
from tf2rl.envs.normalizer import EmpiricalNormalizer
from tensorflow.keras.models import load_model

# train.py
from tf2rl.experiments.trainer import Trainer
from tf2rl.experiments.on_policy_trainer import OnPolicyTrainer
from tf2rl.experiments.irl_trainer import IRLTrainer
from smarts.env.hiway_env import HiWayEnv
from smarts.core.agent import AgentSpec
from smarts.core.agent_interface import AgentInterface
from smarts.core.agent_interface import NeighborhoodVehicles, RGB
from smarts.core.controllers import ActionSpaceType
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

################################## Functions ##################################

#### Load expert trajectories ####
def load_expert_trajectories(filepath):
    filenames = glob.glob(filepath)

    trajectories = []
    for filename in filenames:
        trajectories.append(np.load(filename))

    obses = []
    next_obses = []
    actions = []
    
    for trajectory in trajectories:
        obs = trajectory['obs']
        action = trajectory['act']

        for i in range(obs.shape[0]-1):
            obses.append(obs[i])
            next_obses.append(obs[i+1])
            act = action[i]
            act[0] += random.normalvariate(0, 0.1) # speed
            act[0] = np.clip(act[0], 0, 10)
            act[0] = 2.0 * ((act[0] - 0) / (10 - 0)) - 1.0 # normalize speed
            act[1] += random.normalvariate(0, 0.1) # lane change
            act[1] = np.clip(act[1], -1, 1)
            actions.append(act)
    
    expert_trajs = {'obses': np.array(obses, dtype=np.float32),
                    'next_obses': np.array(next_obses, dtype=np.float32),
                    'actions': np.array(actions, dtype=np.float32)}

    return expert_trajs

# observation space
def observation_adapter(env_obs):
    global states

    new_obs = env_obs.top_down_rgb[1] / 255.0
    states[:, :, 0:3] = states[:, :, 3:6]
    states[:, :, 3:6] = states[:, :, 6:9]
    states[:, :, 6:9] = new_obs

    if env_obs.events.collisions or env_obs.events.reached_goal:
        states = np.zeros(shape=(80, 80, 9))

    return np.array(states, dtype=np.float32)

# reward function
def reward_adapter(env_obs, env_reward):
    progress = env_obs.ego_vehicle_state.speed * 0.1
    goal = 1 if env_obs.events.reached_goal else 0
    crash = -1 if env_obs.events.collisions else 0

    if args.algo == 'value_penalty' or args.algo == 'policy_constraint':
        return goal + crash
    else:
        return 0.01 * progress + goal + crash

# action space
def action_adapter(model_action): 
    speed = model_action[0] # output (-1, 1)
    speed = (speed - (-1)) * (10 - 0) / (1 - (-1)) # scale to (0, 10)
    
    speed = np.clip(speed, 0, 10)
    model_action[1] = np.clip(model_action[1], -1, 1)

    # discretization
    if model_action[1] < -1/3:
        lane = -1
    elif model_action[1] > 1/3:
        lane = 1
    else:
        lane = 0

    return (speed, lane)

# information
def info_adapter(observation, reward, info):
    return info

################################# Initialize ##################################

# Discover and use available GPUs
if tf.config.experimental.list_physical_devices('GPU'):
    for cur_device in tf.config.experimental.list_physical_devices("GPU"):
        print(cur_device)
        tf.config.experimental.set_memory_growth(cur_device, enable=True)

# Environment specs
ACTION_SPACE = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))
OBSERVATION_SPACE = gym.spaces.Box(low=0, high=1, shape=(80, 80, 9))
AGENT_ID = 'Agent-007'
states = np.zeros(shape=(80, 80, 9))

# RL training
parser = Trainer.get_argument()
# parser.add_argument("algo", help="algorithm to run")
# parser.add_argument("scenario", help="scenario to run")
# parser.add_argument("--prior", help="path to the expert prior models", default=None)
args = parser.parse_args()
args.algo = 'value_penalty'
args.scenario = 'left_turn'
args.prior = 'expert_model/left_turn_40/'
args.max_steps = 10e5
args.save_summary_interval = 128
args.use_prioritized_rb = False
args.n_experiments = 10
args.logdir = f'./train_results/{args.scenario}/{args.algo}'

scenario_path = ['scenarios/left_turn']
max_episode_steps = 400

# define agent interface
agent_interface = AgentInterface(
    max_episode_steps=max_episode_steps,
    waypoints=True,
    neighborhood_vehicles=NeighborhoodVehicles(radius=60),
    rgb=RGB(80, 80, 32/80),
    action=ActionSpaceType.LaneWithContinuousSpeed,
)

# define agent specs
agent_spec = AgentSpec(
    interface=agent_interface,
    observation_adapter=observation_adapter,
    reward_adapter=reward_adapter,
    action_adapter=action_adapter,
    info_adapter=info_adapter,
)

class CriticV(tf.keras.Model):
    def __init__(self, state_shape, name='vf'):
        super().__init__(name=name)
        # Learn to extract
        self.conv_layers = [Conv2D(16, 3, strides=3, activation='relu'), Conv2D(64, 3, strides=2, activation='relu'), 
                            Conv2D(128, 3, strides=2, activation='relu'), Conv2D(256, 3, strides=2, activation='relu'), 
                            GlobalAveragePooling2D()]
       
       # Learn to classifly
        self.connect_layers = [Dense(128, activation='relu'), Dense(32, activation='relu')]
        self.out_layer = Dense(1, name="V", activation='linear')

        dummy_state = tf.constant(np.zeros(shape=(1,) + state_shape, dtype=np.float32))
        self(dummy_state)
        #self.summary()

    def call(self, states):
        features = states
        for conv_layer in self.conv_layers:
            features = conv_layer(features) 

        for connect_layer in self.connect_layers:
            features = connect_layer(features)

        values = self.out_layer(features)

        return tf.squeeze(values, axis=1)

class CriticQ(tf.keras.Model):
    def __init__(self, state_shape, action_dim, name='qf'):
        super().__init__(name=name)

        self.conv_layers = [Conv2D(16, 3, strides=3, activation='relu'), Conv2D(64, 3, strides=2, activation='relu'), 
                            Conv2D(128, 3, strides=2, activation='relu'), Conv2D(256, 3, strides=2, activation='relu'),
                            GlobalAveragePooling2D()]
        
        self.act_layers = [Dense(64, activation='relu')]
        self.connect_layers = [Dense(128, activation='relu'), Dense(32, activation='relu')]
        self.out_layer = Dense(1, name="Q", activation='linear')

        dummy_state = tf.constant(np.zeros(shape=(1,) + state_shape, dtype=np.float32))
        dummy_action = tf.constant(np.zeros(shape=[1, action_dim], dtype=np.float32))
        self(dummy_state, dummy_action)
        #self.summary()

    def call(self, states, actions):
        features = states

        for conv_layer in self.conv_layers:
            features = conv_layer(features)

        action = self.act_layers[0](actions)
        features_action = tf.concat([features, action], axis=1)

        for connect_layer in self.connect_layers:
            features_action = connect_layer(features_action)

        values = self.out_layer(features_action)

        return tf.squeeze(values, axis=1)

# Building Agent
class ValuePenalty_SmartAgent(OffPolicyAgent):
    def __init__(
        self,
        state_shape,
        action_dim,
        prior,
        uncertainty='ensemble',
        name="ValuePenalty",
        max_action=1.,
        lr=3e-4,
        tau=5e-3,
        alpha=0.01,
        epsilon=0.5,
        auto_alpha=False,
        n_warmup=int(1e4),
        memory_capacity=int(1e6),
        **kwargs):
        super().__init__(name=name, memory_capacity=memory_capacity, n_warmup=n_warmup, **kwargs)

        self._expert_prior = prior
        self._uncertainty = uncertainty
        self._setup_actor(state_shape, action_dim, lr, max_action)
        self._setup_critic_v(state_shape, lr)
        self._setup_critic_q(state_shape, action_dim, lr)

        # Set hyper-parameters
        self.tau = tau
        self.auto_alpha = auto_alpha
        self.epsilon = epsilon
        self.alpha = alpha

        self.state_ndim = len(state_shape)

    # Initialize actor
    def _setup_actor(self, state_shape, action_dim, lr, max_action=1.0):
        self.actor = ExpertGuidedGaussianActor(state_shape, action_dim, max_action, self._expert_prior, self._uncertainty)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    # Initialize Q
    def _setup_critic_q(self, state_shape, action_dim, lr):
        self.qf1 = CriticQ(state_shape, action_dim, name="qf1")
        self.qf2 = CriticQ(state_shape, action_dim, name="qf2")
        self.qf1_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.qf2_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    # Initialize V
    def _setup_critic_v(self, state_shape, lr):
        self.vf = CriticV(state_shape)
        self.vf_target = CriticV(state_shape)
        update_target_variables(self.vf_target.weights, self.vf.weights, tau=1.0)
        self.vf_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    def get_action(self, state, test=False):
        assert isinstance(state, np.ndarray)
        is_single_state = len(state.shape) == self.state_ndim

        state = np.expand_dims(state, axis=0).astype(np.float32) if is_single_state else state
        action = self._get_action_body(tf.constant(state), test)

        return action.numpy()[0] if is_single_state else action

    @tf.function
    def _get_action_body(self, state, test):
        actions = self.actor(state, test)
        return actions
    
    def update():
        pass
    
# Bulding Environment
env = HiWayEnv(scenarios=scenario_path, agent_specs={AGENT_ID: agent_spec}, headless=True, seed=1)
env.observation_space = OBSERVATION_SPACE
env.action_space = ACTION_SPACE
env.agent_id = AGENT_ID

# Setup train
total_steps = 0
tf.summary.experimental.set_step(total_steps)
episode_steps = 0
episode_return = 0
episode_start_time = time.perf_counter()
n_episode = 0
episode_returns = []
success_log = [0]
best_train = -np.inf

obs = env.reset()
obs = obs[env.agent_id]

agent = ValuePenalty_SmartAgent(state_shape=OBSERVATION_SPACE.shape, action_dim=ACTION_SPACE.high.size, max_action=ACTION_SPACE.high[0], 
                     prior=args.prior, auto_alpha=False, alpha=0.002, memory_capacity=int(2e4), batch_size=32, n_warmup=5000)

# Training
