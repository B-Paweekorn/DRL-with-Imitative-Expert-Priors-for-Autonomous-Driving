# DRL with Imitative Expert Priors for Autonomous Driving

**FRA503 Deep Reinforcement Learning**: Class Project

### Members

- A. Nopparuj   64340500034
- B. Paweekorn  64340500038
- A. Tanakon    64340500062

### References and Acknowledgements

- https://github.com/MCZhi/Expert-Prior-RL
- https://github.com/huawei-noah/SMARTS/tree/v0.4.17
- https://github.com/keiohta/tf2rl

## Setup & Installation

1. Clone the repository
    
    ```bash
    git clone https://github.com/B-Paweekorn/DRL-with-Imitative-Expert-Priors-for-Autonomous-Driving.git
    cd DRL-with-Imitative-Expert-Priors-for-Autonomous-Driving
    ```

2. Install Docker: [Install Docker Engine on Ubuntu](https://docs.docker.com/engine/install/ubuntu/)
    
    Optional (run Docker without sudo): [Linux post-installation steps for Docker Engine](https://docs.docker.com/engine/install/linux-postinstall/)


3. Run SMARTS Docker (must be in the root directory of the project):

    ```bash
    docker run --rm -it -v $PWD:/src -p 8081:8081 --name smarts huaweinoah/smarts:v0.4.17
    ```

> [!TIP]
> If you accidentally close the terminal, you can reattach to the Docker container using the following command:
> ```bash
> docker attach smarts
> ```

> [!NOTE]
> The `--rm` flag will remove the container when it is stopped. If you want to keep the container, remove the `--rm` flag.
> You can restart the container and attach to it using the following command:
> ```bash
> docker start -ai smarts
> ```
> If you want to remove the container, use the following command:
> ```bash
> docker rm smarts
> ```

4. Install the required libraries in the Docker container

    ```bash
    pip install tensorflow-probability==0.10.1 cpprb seaborn==0.11.0
    ```

## Usage

> [!NOTE]
> The following commands should be run in the Docker container.

1. Change directory to `Expert-Prior-RL`
    
    ```bash
    cd Expert-Prior-RL
    ```

2. Run imitation learning to learn the imitative expert policy

    ```bash
    python imitation_learning_uncertainty.py expert_data/left_turn --samples 40
    ```

3. Train RL agent. The available algorithms are sac, value_penalty, policy_constraint, ppo, gail.

    ```bash
    python train.py value_penalty left_turn --prior expert_model/left_turn_40
    ```

4. Evaluate the trained RL agent. Specify the algorithm and the preferred model to evaluate.

    ```bash
    scl run --envision test.py value_penalty left_turn train_results/left_turn/value_penalty/Model/Model_X.h5
    ```

    Open the browser and go to http://localhost:8081 to visualize the evaluation results.

## Background

### 1. Expert Recording
The recorded expert demo files are Numpy zipped archive, each of them contains two Numpy arrays: a series of the expert's actions and a series of the expert's observations.

    - **Action**
        - Array size: (timesteps, 2)
        - Data: [`target_speed`, `lane_change`]
        - `target_speed` is the desired speed of the vehicle within range [0, 10] m/s normalized to [-1, 1].
        - `lane_change` is the desired lane change direction within range of [-1, 1]. The value of -1 indicates the left lane, 0 indicates lane keeping, and 1 indicates the right lane.
        - Lane following controller is used to control the vehicle's steering.

    - **Observation**
        - Array size: (timesteps, 80, 80, 9)
        - Data: 3x RGB images (80x80x3) at `t`, `t-1`, and `t-2` timesteps.
        - The value for each pixel color is within range of [0, 1].

### 2. Imitation Learning
The imitation learning is used to learn the imitative expert policy. The expert data is used to train the model to predict the expert's actions given the expert's observations.

    - The model is a convolutional neural network with 4 convolutional layers and 2 dense layers.
    - The observation-action pairs are used to train the model.

### 3. RL Training

In this section, we will focus on training the RL agent using the imitative expert priors. The training will involve initializing the environment, defining the RL agent, setting up the replay buffer, and running the training loop.
#### 3.1 Setting Up the Environment

First, we'll set up the environment using the HiWayEnv class from the SMARTS library.
 ```bash
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
```
#### 3.2 Agent
 ```bash
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
```
#### 3.3 Training
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

    episode_steps += 1
    episode_return += reward
    total_steps += 1

    obs = next_obs

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
   
