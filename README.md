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
    python DIYexpert_prior.py
    or
    python train.py sac left_turn --prior expert_model/left_turn_40
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

      ![observation](https://github.com/B-Paweekorn/DRL-with-Imitative-Expert-Priors-for-Autonomous-Driving/assets/47713359/3a443b5a-39a0-4f89-a69b-0f3269c4ad44)

### 2. Imitation Learning
The imitation learning is used to learn the imitative expert policy. The expert data is used to train the model to predict the expert's actions given the expert's observations.

![imitation learining](https://github.com/B-Paweekorn/DRL-with-Imitative-Expert-Priors-for-Autonomous-Driving/assets/47713359/4876b383-26dc-4bd4-b3fa-db1f346485c2)

- The model is a convolutional neural network with 4 convolutional layers and 2 dense layers.
- The observation-action pairs are used to train the model.

### 3. RL Training

In this section, we will focus on training the RL agent using the imitative expert priors. The training will involve initializing the environment, defining the RL agent, setting up the replay buffer, and running the training loop.

#### 1. Setting Up the Environment

1. **Observation Space:**
   - Function: `observation_adapter(env_obs)`
   - Steps:
     1. Normalize new observation.
     2. Update the observation history.
     3. Check for collisions or goal completion:
        - If true, reset states.
     4. Return the updated states.

2. **Reward Function:**
   - Function: `reward_adapter(env_obs, env_reward)`
   - Steps:
     1. Calculate progress reward based on speed.
     2. Check if goal is reached.
     3. Check for collisions.
     4. Return rewards based on algorithm type.

3. **Action Space:**
   - Function: `action_adapter(model_action)`
   - Steps:
     1. Scale speed to the range (0, 10).
     2. Clip speed and lane values.
     3. Discretize lane changes.
     4. Return the adjusted action.

#### 2. Agent

1. **Agent Initialization:**
   - Class: `ValuePenalty_SmartAgent`
   - Constructor Parameters: `state_shape`, `action_dim`, `prior`, etc.
   - Steps:
     1. Call superclass constructor.
     2. Set up actor, critic V, and critic Q.
     3. Set hyper-parameters.

2. **Setup Actor:**
   - Function: `_setup_actor(state_shape, action_dim, lr, max_action)`
   - Steps:
     1. Initialize `ExpertGuidedGaussianActor`.
     2. Set up actor optimizer.

3. **Setup Critic Q:**
   - Function: `_setup_critic_q(state_shape, action_dim, lr)`
   - Steps:
     1. Initialize `CriticQ` for Q1 and Q2.
     2. Set up optimizers for Q1 and Q2.

4. **Setup Critic V:**
   - Function: `_setup_critic_v(state_shape, lr)`
   - Steps:
     1. Initialize `CriticV` and its target.
     2. Update target weights.
     3. Set up V optimizer.

5. **Get Action:**
   - Function: `get_action(state, test=False)`
   - Steps:
     1. Check if state is single or batch.
     2. Expand dimensions if necessary.
     3. Get action using `_get_action_body`.
     4. Return action.

6. **Update Agent:**
   - Function: `update(states, actions, next_states, rewards, dones, weights=None)`
   - Steps:
     1. If weights are `None`, set default weights.
     2. Call `_update_body` to perform the update.
     3. Return TD errors.

7. **Update Body:**
   - Function: `_update_body(states, actions, next_states, rewards, dones, weights)`
   - Steps:
     1. Ensure correct shape of rewards and dones.
     2. Compute losses for critic Q, critic V, and actor.
     3. Apply gradients for each loss.
     4. Update target V.
     5. Return update metrics.

#### 3. Training

1. Initialize `total_steps`, `episode_steps`, `episode_return`, and `global_done`.
2. Loop until `total_steps` < `args.max_steps`:
   - If `total_steps` < `agent.n_warmup`:
     - Sample random action from action space.
   - Else:
     - Get action from the agent.
   - Perform environment step with the action.
   - Update `obs`, `reward`, `done`, `info`, and `global_done`.
   - Increment `episode_steps`, `episode_return`, and `total_steps`.
   - If `total_steps` % `agent.update_interval` == 0:
     - Sample from replay buffer.
     - Update agent with sampled data.

## Reward Function Design

- **Value Penalty**

  The reward function for Value Penalty was adjusted so crashing is more significant.

  ```python
  goal = 1 if env_obs.events.reached_goal else 0
  crash = -5 if env_obs.events.collisions else 0

  reward = goal + crash
  ```

## Result

![training result](https://github.com/B-Paweekorn/DRL-with-Imitative-Expert-Priors-for-Autonomous-Driving/assets/47713359/bc84c655-f18e-4f0b-abe0-9d7b04a29bc8)

Our Value penalty perform the best because it is more careful around other cars as the negative reward for crashing is more significant than others. This results in higher success rate and average time taken per episode.

| **algorithm**     | **success rate** | **average time** | **std** |
|-------------------|------------------|------------------|---------|
| **SAC**           | 64.0%            | 11.81 s          | 1.02 s  |
| **Value penalty** | 82.0%            | 11.06 s          | 1.93 s  |
| **Our**           | 96.0%            | 12.12 s          | 2.10 s  |

| Our Value Penalty | Value Penalty | Soft Actor-Critic |
| --- | --- | --- |
| <video src="https://github.com/B-Paweekorn/DRL-with-Imitative-Expert-Priors-for-Autonomous-Driving/assets/47713359/651c950d-7b64-418d-955b-e60730500267"> | <video src="https://github.com/B-Paweekorn/DRL-with-Imitative-Expert-Priors-for-Autonomous-Driving/assets/47713359/f56c4a9a-98f6-4c78-b82a-3317ab204270"> | <video src="https://github.com/B-Paweekorn/DRL-with-Imitative-Expert-Priors-for-Autonomous-Driving/assets/47713359/50b53796-e941-4313-8b87-eddb7b62d2c0"> |

### Conclusion

Even with 96% success rate, this RL model still isn't ideal for real world use. 96% success rate means 4% failure, that is 4 crashes every 100 turns.
