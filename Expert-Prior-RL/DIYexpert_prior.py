################################### Imports ###################################

import os
import sys
import gym
import csv
import time
import glob
import random
import logging
import numpy as np
from PIL import Image
import tensorflow as tf

# expert_prior.py
from tensorflow.keras.layers import Dense, Conv2D, GlobalAveragePooling2D
from tf2rl.algos.policy_base import OffPolicyAgent
from tf2rl.misc.target_update_ops import update_target_variables
from tf2rl.policies.tfp_gaussian_actor import ExpertGuidedGaussianActor

# tf2rl trainer.py
from tf2rl.misc.get_replay_buffer import get_replay_buffer
from tf2rl.misc.prepare_output_dir import prepare_output_dir
from tf2rl.misc.initialize_logger import initialize_logger

# train.py
from tf2rl.experiments.trainer import Trainer
from smarts.env.hiway_env import HiWayEnv
from smarts.core.agent import AgentSpec
from smarts.core.agent_interface import AgentInterface
from smarts.core.agent_interface import NeighborhoodVehicles, RGB
from smarts.core.controllers import ActionSpaceType

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
        obs = trajectory["obs"]
        action = trajectory["act"]

        for i in range(obs.shape[0] - 1):
            obses.append(obs[i])
            next_obses.append(obs[i + 1])
            act = action[i]
            act[0] += random.normalvariate(0, 0.1)  # speed
            act[0] = np.clip(act[0], 0, 10)
            act[0] = 2.0 * ((act[0] - 0) / (10 - 0)) - 1.0  # normalize speed
            act[1] += random.normalvariate(0, 0.1)  # lane change
            act[1] = np.clip(act[1], -1, 1)
            actions.append(act)

    expert_trajs = {
        "obses": np.array(obses, dtype=np.float32),
        "next_obses": np.array(next_obses, dtype=np.float32),
        "actions": np.array(actions, dtype=np.float32),
    }

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
    global global_done

    progress = env_obs.ego_vehicle_state.speed * 0.1
    goal = 1 if env_obs.events.reached_goal else 0
    crash = -1 if env_obs.events.collisions else 0
    
    if args.algo == "value_penalty" or args.algo == "policy_constraint":
        return goal + crash
        # return (goal * 6 + crash * 3) - 2 * global_done
    else:
        return 0.01 * progress + goal + crash


# action space
def action_adapter(model_action):
    speed = model_action[0]  # output (-1, 1)
    speed = (speed - (-1)) * (10 - 0) / (1 - (-1))  # scale to (0, 10)

    speed = np.clip(speed, 0, 10)
    model_action[1] = np.clip(model_action[1], -1, 1)

    # discretization
    if model_action[1] < -1 / 3:
        lane = -1
    elif model_action[1] > 1 / 3:
        lane = 1
    else:
        lane = 0

    return (speed, lane)


# information
def info_adapter(observation, reward, info):
    return info


################################# Initialize ##################################

# Discover and use available GPUs
if tf.config.experimental.list_physical_devices("GPU"):
    for cur_device in tf.config.experimental.list_physical_devices("GPU"):
        print(cur_device)
        tf.config.experimental.set_memory_growth(cur_device, enable=True)

# Environment specs
ACTION_SPACE = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))
OBSERVATION_SPACE = gym.spaces.Box(low=0, high=1, shape=(80, 80, 9))
AGENT_ID = "Agent-007"
states = np.zeros(shape=(80, 80, 9))

global_done = 0

# Set arguments
parser = Trainer.get_argument()
if True:
    # parser.add_argument("algo", help="algorithm to run")
    # parser.add_argument("scenario", help="scenario to run")
    # parser.add_argument("--prior", help="path to the expert prior models", default=None)
    args = parser.parse_args()
    args.algo = "value_penalty"
    args.scenario = "left_turn"
    args.prior = "expert_model/left_turn_40/"
    args.max_steps = 10e4
    args.save_summary_interval = 128
    args.use_prioritized_rb = False
    args.n_experiments = 10
    args.logdir = f"./train_results/{args.scenario}/{args.algo}"

scenario_path = ["scenarios/left_turn"]
max_episode_steps = 400

# define agent interface
agent_interface = AgentInterface(
    max_episode_steps=max_episode_steps,
    waypoints=True,
    neighborhood_vehicles=NeighborhoodVehicles(radius=60),
    rgb=RGB(80, 80, 32 / 80),
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
    def __init__(self, state_shape, name="vf"):
        super().__init__(name=name)
        # Learn to extract
        self.conv_layers = [
            Conv2D(16, 3, strides=3, activation="relu"),
            Conv2D(64, 3, strides=2, activation="relu"),
            Conv2D(128, 3, strides=2, activation="relu"),
            Conv2D(256, 3, strides=2, activation="relu"),
            GlobalAveragePooling2D(),
        ]

        # Learn to classifly
        self.connect_layers = [
            Dense(128, activation="relu"),
            Dense(32, activation="relu"),
        ]
        self.out_layer = Dense(1, name="V", activation="linear")

        dummy_state = tf.constant(np.zeros(shape=(1,) + state_shape, dtype=np.float32))
        self(dummy_state) 
        # self.summary()

    def call(self, states):
        features = states
        for conv_layer in self.conv_layers:
            features = conv_layer(features)

        for connect_layer in self.connect_layers:
            features = connect_layer(features)

        values = self.out_layer(features)

        return tf.squeeze(values, axis=1)


class CriticQ(tf.keras.Model):
    def __init__(self, state_shape, action_dim, name="qf"):
        super().__init__(name=name)

        self.conv_layers = [
            Conv2D(16, 3, strides=3, activation="relu"),
            Conv2D(64, 3, strides=2, activation="relu"),
            Conv2D(128, 3, strides=2, activation="relu"),
            Conv2D(256, 3, strides=2, activation="relu"),
            GlobalAveragePooling2D(),
        ]

        self.act_layers = [Dense(64, activation="relu")]
        self.connect_layers = [
            Dense(128, activation="relu"),
            Dense(32, activation="relu"),
        ]
        self.out_layer = Dense(1, name="Q", activation="linear")

        dummy_state = tf.constant(np.zeros(shape=(1,) + state_shape, dtype=np.float32))
        dummy_action = tf.constant(np.zeros(shape=[1, action_dim], dtype=np.float32))
        self(dummy_state, dummy_action)
        # self.summary()

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


############################### Building Agent ################################


class ValuePenalty_SmartAgent(OffPolicyAgent):
    def __init__(
        self,
        state_shape,
        action_dim,
        prior,
        uncertainty="ensemble",
        name="ValuePenalty",
        max_action=1.0,
        lr=5e-4,
        tau=5e-3,
        alpha=0.01,
        epsilon=0.5,
        auto_alpha=False,
        n_warmup=int(1e4),
        memory_capacity=int(1e6),
        **kwargs,
    ):
        super().__init__(
            name=name, memory_capacity=memory_capacity, n_warmup=n_warmup, **kwargs
        )

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
        self.actor = ExpertGuidedGaussianActor(
            state_shape, action_dim, max_action, self._expert_prior, self._uncertainty
        )
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

        state = (
            np.expand_dims(state, axis=0).astype(np.float32)
            if is_single_state
            else state
        )
        action = self._get_action_body(tf.constant(state), test)

        return action.numpy()[0] if is_single_state else action

    @tf.function
    def _get_action_body(self, state, test):
        actions, log_pis, entropy, std, expert_log_pis, kl = self.actor(state, test)
        return actions

    def update(self, states, actions, next_states, rewards, dones, weights=None):
        if weights is None:
            weights = np.ones_like(rewards)

        (
            td_errors,
            actor_loss,
            vf_loss,
            qf_loss,
            q_value,
            logp_min,
            logp_max,
            logp_mean,
            entropy_mean,
            kl_mean,
            std_mean,
            std_max,
            std_min,
        ) = self._update_body(states, actions, next_states, rewards, dones, weights)

        tf.summary.scalar(name=self.policy_name + "/actor_loss", data=actor_loss)
        tf.summary.scalar(name=self.policy_name + "/critic_V_loss", data=vf_loss)
        tf.summary.scalar(name=self.policy_name + "/critic_Q_loss", data=qf_loss)
        tf.summary.scalar(name=self.policy_name + "/Q_value", data=q_value)
        tf.summary.scalar(name=self.policy_name + "/logp_min", data=logp_min)
        tf.summary.scalar(name=self.policy_name + "/logp_max", data=logp_max)
        tf.summary.scalar(name=self.policy_name + "/logp_mean", data=logp_mean)
        tf.summary.scalar(name=self.policy_name + "/entropy", data=entropy_mean)
        tf.summary.scalar(name=self.policy_name + "/kl_divergence", data=kl_mean)
        tf.summary.scalar(name=self.policy_name + "/expert_std_mean", data=std_mean)
        tf.summary.scalar(name=self.policy_name + "/expert_std_max", data=std_max)
        tf.summary.scalar(name=self.policy_name + "/expert_std_min", data=std_min)
        tf.summary.scalar(name=self.policy_name + "/alpha", data=self.alpha)

        return td_errors

    @tf.function
    def _update_body(self, states, actions, next_states, rewards, dones, weights):
        with tf.device(self.device):
            assert len(dones.shape) == 2
            assert len(rewards.shape) == 2
            rewards = tf.squeeze(rewards, axis=1)
            dones = tf.squeeze(dones, axis=1)

            not_dones = 1.0 - tf.cast(dones, dtype=tf.float32)

            with tf.GradientTape(persistent=True) as tape:
                # Compute loss of critic Q
                current_q1 = self.qf1(states, actions)
                current_q2 = self.qf2(states, actions)
                next_v_target = self.vf_target(next_states)

                target_q = tf.stop_gradient(
                    rewards + not_dones * self.discount * next_v_target
                )

                td_loss_q1 = tf.reduce_mean((target_q - current_q1) ** 2)
                td_loss_q2 = tf.reduce_mean((target_q - current_q2) ** 2)

                # Compute loss of critic V
                current_v = self.vf(states)

                sample_actions, logp, entropy, std, _, kl = self.actor(
                    states
                )  # Resample actions to update V
                current_q1 = self.qf1(states, sample_actions)
                current_q2 = self.qf2(states, sample_actions)
                current_min_q = tf.minimum(current_q1, current_q2)

                if self.auto_alpha:
                    target_v = tf.stop_gradient(current_min_q)
                else:
                    target_v = tf.stop_gradient(current_min_q - self.alpha * kl)

                td_errors = target_v - current_v
                td_loss_v = tf.reduce_mean(td_errors**2)

                policy_loss = tf.reduce_mean(self.alpha * kl - current_min_q)

            # Critic Q1 loss
            q1_grad = tape.gradient(td_loss_q1, self.qf1.trainable_variables)
            self.qf1_optimizer.apply_gradients(
                zip(q1_grad, self.qf1.trainable_variables)
            )

            # Critic Q2 loss
            q2_grad = tape.gradient(td_loss_q2, self.qf2.trainable_variables)
            self.qf2_optimizer.apply_gradients(
                zip(q2_grad, self.qf2.trainable_variables)
            )

            # Critic V loss
            vf_grad = tape.gradient(td_loss_v, self.vf.trainable_variables)
            self.vf_optimizer.apply_gradients(zip(vf_grad, self.vf.trainable_variables))
            # Update Target V
            update_target_variables(self.vf_target.weights, self.vf.weights, self.tau)

            # Actor loss
            actor_grad = tape.gradient(policy_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(
                zip(actor_grad, self.actor.trainable_variables)
            )

            del tape

        return (
            td_errors,
            policy_loss,
            td_loss_v,
            td_loss_q1,
            tf.reduce_mean(current_min_q),
            tf.reduce_min(logp),
            tf.reduce_max(logp),
            tf.reduce_mean(logp),
            tf.reduce_mean(entropy),
            tf.reduce_mean(kl),
            tf.reduce_mean(std),
            tf.reduce_max(std),
            tf.reduce_min(std),
        )



# Bulding Environment
env = HiWayEnv(
    scenarios=scenario_path, agent_specs={AGENT_ID: agent_spec}, headless=True, seed=1
)
env.observation_space = OBSERVATION_SPACE
env.action_space = ACTION_SPACE
env.agent_id = AGENT_ID

############################### Setup Training ################################

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

agent = ValuePenalty_SmartAgent(
    state_shape=OBSERVATION_SPACE.shape,
    action_dim=ACTION_SPACE.high.size,
    max_action=ACTION_SPACE.high[0],
    prior=args.prior,
    auto_alpha=False,
    alpha=0.002,
    memory_capacity=int(2e4),
    batch_size=32,
    n_warmup=5000,
)

replay_buffer = get_replay_buffer(
    agent, env, args.use_prioritized_rb, args.use_nstep_rb, args.n_step
)

# prepare log directory
_output_dir = prepare_output_dir(
    args=args,
    user_specified_dir=args.logdir,
    suffix="{}_{}".format(agent.policy_name, args.dir_suffix),
)

_episode_max_steps = (
    args.episode_max_steps if args.episode_max_steps is not None else args.max_steps
)

logger = initialize_logger(
    logging_level=logging.getLevelName(args.logging_level), output_dir=_output_dir
)

# prepare TensorBoard output
writer = tf.summary.create_file_writer(_output_dir)
writer.set_as_default()

# Save and restore model
_checkpoint = tf.train.Checkpoint(policy=agent)
checkpoint_manager = tf.train.CheckpointManager(
    _checkpoint, directory=_output_dir, max_to_keep=5
)
model_dir = args.model_dir
if model_dir is not None:
    assert os.path.isdir(model_dir)
    _latest_path_ckpt = tf.train.latest_checkpoint(model_dir)
    _checkpoint.restore(_latest_path_ckpt)
    logger.info("Restored {}".format(_latest_path_ckpt))


################################## Training ###################################

while total_steps < args.max_steps:
    # print(n_episode)
    if total_steps < agent.n_warmup:
        action = env.action_space.sample()
    else:
        action = agent.get_action(obs)
    next_obs, reward, done, info = env.step({env.agent_id: action})
    next_obs = next_obs[env.agent_id]

    reward = reward[env.agent_id]
    done = done[env.agent_id]
    info = info[env.agent_id]
    global_done = done

    if args.show_progress:
        obs_tensor = tf.expand_dims(obs, axis=0)
        # agent distribution
        agent_dist = agent.actor._compute_dist(obs_tensor)

    episode_steps += 1
    episode_return += reward
    total_steps += 1
    tf.summary.experimental.set_step(total_steps)

    # if the episode is finished
    done_flag = done
    if hasattr(env, "_max_episode_steps") and episode_steps == env._max_episode_steps:
        done_flag = False

    replay_buffer.add(
        obs=obs, act=action, next_obs=next_obs, rew=reward, done=done_flag
    )
    obs = next_obs

    # add to training log
    if total_steps % 5 == 0:
        success = np.sum(success_log[-20:]) / 20
        with open(_output_dir + "/training_log.csv", "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    n_episode,
                    total_steps,
                    episode_returns[n_episode - 1] if episode_returns else -1,
                    success,
                    episode_steps,
                ]
            )

    # end of a episode
    if done or episode_steps == _episode_max_steps:
        # if task is successful
        success_log.append(1 if info["env_obs"].events.reached_goal else 0)

        # reset env
        replay_buffer.on_episode_end()
        obs = env.reset()
        obs = obs[env.agent_id]

        # display info
        n_episode += 1
        fps = episode_steps / (time.perf_counter() - episode_start_time)
        success = np.sum(success_log[-20:]) / 20

        logger.info(
            "Total Episode: {0: 5} Steps: {1: 7} Episode Steps: {2: 5} Return: {3: 5.4f} FPS: {4:5.2f}".format(
                n_episode, total_steps, episode_steps, episode_return, fps
            )
        )

        print("rates", success)

        tf.summary.scalar(name="Common/training_return", data=episode_return)
        tf.summary.scalar(name="Common/training_success", data=success)
        tf.summary.scalar(name="Common/training_episode_length", data=episode_steps)

        # reset variables
        episode_returns.append(episode_return)
        episode_steps = 0
        episode_return = 0
        episode_start_time = time.perf_counter()

        # save policy model
        if n_episode > 20 and np.mean(episode_returns[-20:]) >= best_train:
            best_train = np.mean(episode_returns[-20:])
            agent.actor.network.save(
                "{}/Model/Model_{}_{:.4f}.h5".format(args.logdir, n_episode, best_train)
            )

    if total_steps < agent.n_warmup:
        continue
    
    # main update
    if total_steps % agent.update_interval == 0:
        samples = replay_buffer.sample(agent.batch_size)

        with tf.summary.record_if(total_steps % args.save_summary_interval == 0):
            agent.update(
                samples["obs"],
                samples["act"],
                samples["next_obs"],
                samples["rew"],
                np.array(samples["done"], dtype=np.float32),
                None if not args.use_prioritized_rb else samples["weights"],
            )

    # save checkpoint
    if total_steps % args.save_model_interval == 0:
        checkpoint_manager.save()

tf.summary.flush()
