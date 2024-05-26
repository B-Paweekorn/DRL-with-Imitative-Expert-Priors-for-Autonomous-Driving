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

1. Expert Recording

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

2. Imitation Learning

    The imitation learning is used to learn the imitative expert policy. The expert data is used to train the model to predict the expert's actions given the expert's observations.

    - The model is a convolutional neural network with 4 convolutional layers and 2 dense layers.
    - The observation-action pairs are used to train the model.

3. RL Agent Training
