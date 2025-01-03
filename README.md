# Reinforcement-Learning-OpenAI-gym
Adaptive Reinforcement Learning on CartPole (discrete) and CarRacing (continuous) environments with Noise Strategies

## **Description**  
This project investigates how reinforcement learning (RL) agents can adapt to real-world uncertainties using multiple noise strategies. It implements a noise injection framework for RL environments and evaluates advanced RL architectures across discrete and continuous control tasks.  

Key contributions include:  
- A framework for concurrent noise components (friction, wind, and Gaussian noise).  
- Evaluation of four noise strategies: curriculum learning, stochastic, annealing, and dynamic randomization.  
- Comparative performance analysis of RL architectures: Dueling DQN, Double DQN, Noisy Dueling DQN, and Distributional Dueling DQN.  
- Experiments conducted on CartPole (discrete) and CarRacing (continuous) environments.  

---

## **Key Features**  
- Noise injection framework mimicking real-world uncertainties.  
- Four noise strategies to simulate dynamic and stochastic conditions.  
- Extensive comparison of RL agents to identify the best training and architectural approaches.  
- Transferability analysis from low-noise to high-noise environments.  

---

## **Installation**  
Follow these steps to set up the project locally:  

```bash  
# Clone the repository  
git clone https://github.com/yourusername/adaptive-rl-noise-strategies.git  

# Navigate to the project directory  
cd adaptive-rl-noise-strategies  

# Install required dependencies  
pip install -r requirements.txt  
