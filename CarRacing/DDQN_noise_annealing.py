
# Environment imports
import random
import numpy as np
import gym

import cv2
from scipy import stats

# Tensorflow training imports
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# Eager execution to speed up training speeds (A LOT!)
# tf.compat.v1.disable_eager_execution()

# Training monitoring imports
import datetime, os
from tqdm import tqdm
import time
from plot_results import plot_rewards
from env_testing import RandomizedCarRacingTesting


############################## CONFIGURATION ##################################
# Prevent tensorflow from allocating the all of GPU memory
# From: https://stackoverflow.com/questions/34199233/how-to-prevent-tensorflow-from-allocating-the-totality-of-a-gpu-memory
GPUs = tf.config.experimental.list_physical_devices('GPU')
for gpu in GPUs:
    tf.config.experimental.set_memory_growth( gpu, True )   # set memory growth option

# Where are models saved? How frequently e.g. every x1 episode?
USERNAME                = "Eajun"
MODEL_TYPE              = "DDQN2"
NOISE_TYPE              = "ANNEALING"
TIMESTAMP               = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
MODEL_DIR               = f"./model/{USERNAME}/{MODEL_TYPE}/{NOISE_TYPE}/{TIMESTAMP}/"

# Setup Reward Dir
REWARD_DIR              = f"rewards/{USERNAME}/{MODEL_TYPE}/{NOISE_TYPE}/{TIMESTAMP}/"

# Training params
RENDER                  = False
PLOT_RESULTS            = False     # plotting reward and epsilon vs epsiode (graphically) NOTE: THIS WILL PAUSE TRAINING AT PLOT EPISODE!
EPISODES                = 501      # training episodes
SAVE_TRAINING_FREQUENCY = 100       # save model every n episodes
SKIP_FRAMES             = 2         # skip n frames between batches
TARGET_UPDATE_STEPS     = 5         # update target action value network every n EPISODES
MAX_PENALTY             = -5        # min score before env reset
BATCH_SIZE              = 10        # number for batch fitting
CONSECUTIVE_NEG_REWARD  = 20        # number of consecutive negative rewards before terminating episode
STEPS_ON_GRASS          = 5         # How many steps can car be on grass for (steps == states)

# Testing params
PRETRAINED_PATH         = "model/Eajun/DDQN2/ANNEALING/20241206-014005/episode_400.weights.h5"
TEST                    = True     # true = testing, false = training


############################## MAIN CODE BODY ##################################
import numpy as np
import gym
import math

class RandomizedCarRacingAnnealing(gym.Wrapper):
    def __init__(self, env, noise_decrease_rate=0.01, max_noise=1.0, min_noise=0.1, max_episodes=500):
        super().__init__(env)
        self.env = env
        self.noise_decrease_rate = noise_decrease_rate  # Rate at which noise decreases
        self.max_noise = max_noise  # Starting maximum noise
        self.min_noise = min_noise  # Minimum noise threshold
        self.max_episodes = max_episodes  # Total episodes to scale noise across
        self.episode_count = 0  # Track the current episode count
        self.noise_stage = 0  # Track which noise component is active
        self.noise_enabled = [True, True, True]  # Enable all noise components (Wind, Friction, Gaussian)

        # Initialize dynamic noise parameters
        self.friction = self.max_noise  # Affects braking
        self.wind_force = self.max_noise  # Affects steering
        self.gaussian_noise_std = self.max_noise  # Affects observation noise

        self.randomize_environment()

    def get_annealing_range(self, base_min, base_max, noise_type="generic"):
        """Calculate the range for the noise, annealing over episodes."""
        progress = min(self.episode_count / self.max_episodes, 1.0)  # Determine progress as fraction of total episodes

        # Gradually shrink the range from (base_max, base_max) to (base_min, base_min)
        min_range = base_min + (1 - progress) * (base_max - base_min)
        max_range = base_max - progress * (base_max - base_min)

        # Apply specific constraints for each noise type
        if noise_type == "wind":
            min_range = np.clip(min_range, -0.4, 0.4)
            max_range = np.clip(max_range, -0.4, 0.4)
        elif noise_type == "friction":
            min_range = np.clip(min_range, 0.05, 1)
            max_range = np.clip(max_range, 0.05, 1)
        elif noise_type == "gaussian":
            min_range = np.clip(min_range, 0.0, 0.5)
            max_range = np.clip(max_range, 0.0, 0.5)

        return min_range, max_range

    def randomize_environment(self):
        """Randomize environment parameters at the start of an episode."""
        self.env.unwrapped.gravity = np.random.uniform(8.0, 10.0)  # Adjust gravity
        print(f"Random Gravity Set: {self.env.unwrapped.gravity}")

    def apply_friction(self, brake_action):
        """Modify braking based on friction."""
        min_friction, max_friction = self.get_annealing_range(self.min_noise, self.max_noise, "friction")
        self.friction = np.random.uniform(min_friction, max_friction)  # Update friction based on annealing range
        print(f"Friction Force Applied: {self.friction}")
        modified_brake = brake_action * (1 - self.friction)  # Reduces brake effectiveness
        return np.clip(modified_brake, 0.0, 1.0)

    def apply_wind(self, steering_action):
        """Modify steering based on wind force."""
        min_wind, max_wind = self.get_annealing_range(-self.max_noise, self.max_noise, "wind")
        self.wind_force = np.random.uniform(min_wind, max_wind)  # Update wind force based on annealing range
        print(f"Wind Force Applied: {self.wind_force}")
        modified_steering = steering_action + self.wind_force
        return np.clip(modified_steering, -1.0, 1.0)

    def add_gaussian_noise(self, observation):
        """Add Gaussian noise to the observation."""
        min_gauss, max_gauss = self.get_annealing_range(0.0, self.max_noise, "gaussian")
        self.gaussian_noise_std = np.random.uniform(min_gauss, max_gauss)  # Update noise standard deviation
        print(f"Gaussian Force Applied: {self.gaussian_noise_std}")
        noise = np.random.normal(0, self.gaussian_noise_std, size=observation.shape)
        return observation + noise

    def step(self, action):
        """
        Modify actions dynamically:
        - Apply wind to steering
        - Apply friction to braking
        - Add Gaussian noise to observations
        """
        steering, gas, brake = action

        # Modify actions based on enabled noise
        modified_steering = self.apply_wind(steering) if self.noise_enabled[0] else steering
        modified_brake = self.apply_friction(brake) if self.noise_enabled[1] else brake

        # Create modified action
        modified_action = [modified_steering, gas, modified_brake]

        # Take a step in the environment
        observation, reward, terminated, truncated, info = self.env.step(modified_action)

        # Add Gaussian noise to the observation
        noisy_observation = self.add_gaussian_noise(observation) if self.noise_enabled[2] else observation

        # Ensure the state is in the correct type (uint8) before returning it
        noisy_observation = np.clip(noisy_observation, 0, 255).astype(np.uint8)

        return noisy_observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Randomize environment parameters and reset the environment."""
        self.randomize_environment()
        observation, info = self.env.reset(**kwargs)  # Capture both observation and info

        # Increment the episode count
        self.episode_count += 1

        # Ensure the observation is in the correct type (uint8) before returning it
        observation = np.clip(observation, 0, 255).astype(np.uint8)
        return observation, info


class DDQN_Agent:
    def __init__(   self, 
                    action_space    = [
                    (-1, 1, 0.2), (0, 1, 0.2), (1, 1, 0.2), #            Action Space Structure
                    (-1, 1,   0), (0, 1,   0), (1, 1,   0), #           (Steering, Gas, Break)
                    (-1, 0, 0.2), (0, 0, 0.2), (1, 0, 0.2), # Range       -1~1     0~`1`   0~1
                    (-1, 0,   0), (0, 0,   0), (1, 0,   0)],
                    memory_size     = 10000,     # threshold memory limit for replay buffer
                    gamma           = 0.95,      # discount rate
                    epsilon         = 1.0,       # exploration rate
                    epsilon_min     = 0.1,       # used by Atari
                    epsilon_decay   = 0.9999,
                    learning_rate   = 0.001
                ):
        
        self.action_space    = action_space
        self.D               = deque( maxlen=memory_size )
        self.gamma           = gamma
        self.epsilon         = epsilon
        self.epsilon_min     = epsilon_min
        self.epsilon_decay   = epsilon_decay
        self.learning_rate   = learning_rate
        
        # clone the action value network to make target action value network
        self.model           = self.build_model()
        self.target_model    = tf.keras.models.clone_model( self.model )

    def build_model( self ):
        """Sequential Neural Net with x2 Conv layers, x2 Dense layers using RELU and Huber Loss"""
        model = Sequential()
        model.add(Conv2D(filters=6, kernel_size=(7, 7), strides=3, activation='relu', input_shape=(96, 96, 1)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=12, kernel_size=(4, 4), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(216, activation='relu'))
        # model.add(Dense(54, activation='relu'))
        model.add(Dense(len(self.action_space), activation=None))
        model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=self.learning_rate, epsilon=1e-7))
        return model

    def update_model( self ):
        """Update Target Action Value Network to be equal to Action Value Network"""
        self.target_model.set_weights( self.model.get_weights() )
    
    def store_transition( self, state, action, reward, new_state, done ):
        """Store transition in the replay memory (for replay buffer)."""
        self.D.append( (state, action, reward, new_state, done) )

    def choose_action( self, state, best=False):
        """Take state input and use latest target model to make prediction on best next action; choose it!"""
        state = np.expand_dims(state, axis=0)
        actionIDX = np.argmax( self.model.predict(state)[0] )

        # return best action if defined
        if best: return self.action_space[ actionIDX ]

        # epsilon chance to choose random action
        if stats.bernoulli( self.epsilon ).rvs():
            actionIDX =  random.randrange( len(self.action_space) )
        return self.action_space[ actionIDX ]

    def batch_priority( self ):
        """Implementation of prioritised replay where most recent states take highest priority."""
        options = list( range(1,len(self.D)+1) )
        minibatch = []
        for _ in range( BATCH_SIZE ):
            # update probs and create new distribution
            total = len( options ) * (len( options ) + 1) // 2
            prob_dist = [ i/total for i in range(1, len(options)+1) ]

            choice = np.random.choice( options, 1, p = prob_dist )[0]
            del options[ options.index(choice) ]
            minibatch.append( self.D[ choice-1 ] )
        return minibatch

    def experience_replay( self ):
        """Use experience_replay with batch fitting and epsilon decay."""
        if len( self.D ) >= BATCH_SIZE:
            # batch sample randomly
            # minibatch = random.sample( self.D, BATCH_SIZE )

            # select batch based on prioritied replay approach
            minibatch = self.batch_priority()

            # experience replay
            train_state = []
            train_target = []
            for state, action, reward, next_state, done in minibatch:
                target = self.model.predict(np.expand_dims(state, axis=0))[0]
                if done:
                    target[ self.action_space.index(action) ] = reward
                else:
                    ############ Double Deep Q Learning Here! #############
                    # get index of action value network prediction for best action at next state
                    t = self.model.predict(np.expand_dims(next_state, axis=0))[0]
                    t_index = np.where(t == np.amax(t))[0][0]

                    # get target network prediction for next state, then use index calc'd above to
                    # update Q action value network
                    target_t = self.target_model.predict(np.expand_dims(next_state, axis=0))[0]
                    target[ self.action_space.index(action) ] = reward + self.gamma * target_t[ t_index ]

                train_state.append(state)
                train_target.append(target)

            # batch fitting
            self.model.fit( np.array(train_state), np.array(train_target), epochs=1, verbose=0 )
            
            # epsilon decay
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def save( self, name, data ):
        """Save model and rewards list to appropriate dir, defined at start of code."""
        # saving model
        if not os.path.exists( MODEL_DIR ):
             os.makedirs( MODEL_DIR )
        self.target_model.save_weights( MODEL_DIR + name + ".h5" )

        # saving results
        if not os.path.exists( REWARD_DIR ):
             os.makedirs( REWARD_DIR )
        np.savetxt(f"{REWARD_DIR}" + name + ".csv", data, delimiter=",")
        
        # plotting results
        if PLOT_RESULTS: plot_rewards( f"{REWARD_DIR}" + name + ".csv", "Annealing" )

    def load( self, name ):
        """Load previously trained model weights."""
        self.model.load_weights( name )
        self.model.set_weights( self.model.get_weights() )




def convert_greyscale( state ):
    """Take input state and convert to greyscale. Check if road is visible in frame."""
    global on_grass_counter

    x, y, _ = state.shape
    cropped = state[ 0:int( 0.85*y ) , 0:x ]
    mask = cv2.inRange( cropped,  np.array([100, 100, 100]),  # dark_grey
                                  np.array([150, 150, 150]))  # light_grey

    # Create greyscale then normalise array to reduce complexity for neural network
    gray = cv2.cvtColor( state, cv2.COLOR_BGR2GRAY )
    gray = gray.astype(float)
    gray_normalised = gray / 255.0

    # check if car is on grass
    xc = int(x / 2)
    grass_mask = cv2.inRange(   state[67:76 , xc-2:xc+2],
                                np.array([50, 180, 0]),
                                np.array([150, 255, 255]))

    # If on grass for x5 frames or more then trigger True!
    on_grass_counter = on_grass_counter+1 if np.any(grass_mask==255) and "on_grass_counter" in globals() else 0
    if on_grass_counter > STEPS_ON_GRASS:
        on_grass = True
        on_grass_counter = 0
    else: on_grass = False

    # returns [ greyscale image, T/F of if road is visible, is car on grass bool ]
    return [ np.expand_dims( gray_normalised, axis=2 ), np.any(mask== 255), on_grass ]

def train_agent(agent: DDQN_Agent, env: gym.make, episodes: int):
    """Train agent with experience replay, batch fitting and using a cropped greyscale input image."""
    episode_rewards = []  # List to store rewards per episode

    
    for episode in tqdm(range(episodes)):
        print(f"[INFO]: Starting Episode {episode}")
        
        state_colour, _ = env.reset()  # Extract only the state (image)
        state_grey, can_see_road, car_on_grass = convert_greyscale(state_colour)

        sum_reward = 0
        step = 0
        done = False
        while not done and sum_reward > MAX_PENALTY and can_see_road:  # Add other conditions if needed
            # choose action to take next
            action = agent.choose_action(state_grey)

            # take action and observe new state, reward and if terminal.
            reward = 0
            for _ in range(SKIP_FRAMES + 1):
                new_state_colour, r, done, truncated, info = env.step(action)
                reward += r  # Accumulate rewards over skipped frames

                # Convert reward to scalar
                reward = float(np.sum(reward)) if isinstance(reward, np.ndarray) else float(reward)

                # Render if specified, break if terminal
                if RENDER: env.render()
                if done: break

            # Count number of negative rewards collected sequentially
            repeat_neg_reward = repeat_neg_reward + 1 if reward < 0 else 0
            if repeat_neg_reward >= CONSECUTIVE_NEG_REWARD:
                break


            # convert to greyscale for NN
            new_state_grey, can_see_road, car_on_grass = convert_greyscale(new_state_colour)

            # clip reward to 1
            reward = np.clip(reward, a_max=1, a_min=-10)

            # store transition states for experience replay
            agent.store_transition(state_grey, action, reward, new_state_grey, done)

            # do experience replay training with a batch of data
            agent.experience_replay()

            # update params for next loop
            state_grey = new_state_grey
            sum_reward += reward
            step += 1

        # Store the reward of the episode
        episode_rewards.append(sum_reward)

        # update target action value network every N steps ( to equal action value network)
        if episode % TARGET_UPDATE_STEPS == 0:
            agent.update_model()

        # save training progress at intervals
        if episode % SAVE_TRAINING_FREQUENCY == 0:
            agent.save(f"episode_{episode}.weights", data=episode_rewards)

    env.close()

def test_agent( agent : DDQN_Agent, env : gym.make, model : str, testnum=10 ):
    """Test a pretrained model and print out run rewards and total time taken. Quit with ctrl+c."""
    # Load agent model
    agent.load( model )
    run_rewards = []
    for test in range(testnum):
        state_colour, info = env.reset() 
        state_grey, _, _ = convert_greyscale( state_colour )

        done = False
        sum_reward = 0.0
        t1 = time.time()  # Trial timer
        while sum_reward > MAX_PENALTY and not done:

            # choose action to take next
            action = agent.choose_action( state_grey, best=True )
            
            # take action and observe new state, reward and if terminal
            new_state_colour, reward, done, _, _ = env.step( action )

            # render if user has specified
            if RENDER: env.render()

            # Count number of negative rewards collected sequentially, if reward non-negative, restart counting
            repeat_neg_reward = repeat_neg_reward+1 if reward < 0 else 0
            if repeat_neg_reward >= 300: break

            # convert to greyscale for NN
            new_state_grey, _, _ = convert_greyscale( new_state_colour )

            # update state
            state_grey = new_state_grey
            sum_reward += reward

        t1 = time.time()-t1
        run_rewards.append( [sum_reward, np.nan, t1, np.nan, np.nan, np.nan] )
        print(f"[INFO]: Run {test} | Run Reward: ", sum_reward, " | Time:", "%0.2fs."%t1 )

    # calculate useful statistics
    rr = [ i[0] for i in run_rewards ]
    rt = [ i[2] for i in run_rewards ]

    r_max = max(rr)
    r_min = min(rr)
    r_std_dev = np.std( rr )
    r_avg = np.mean(rr)
    t_avg = np.mean(rt)
    
    run_rewards.append( [r_avg, np.nan, t_avg, r_max, r_min, r_std_dev] )    # STORE AVG RESULTS AS LAST ENTRY!
    print(f"[INFO]: Runs {testnum} | Avg Run Reward: ", "%0.2f"%r_avg, "| Avg Time:", "%0.2fs"%t_avg,
            f" | Max: {r_max} | Min: {r_min} | Std Dev: {r_std_dev}" )

    # saving test results
    if not os.path.exists( f"test_{REWARD_DIR}" ):
            os.makedirs( f"test_{REWARD_DIR}" )
    path = f"test_{REWARD_DIR}" +  PRETRAINED_PATH.split('/')[-1][:-3] + "_run_rewards.csv"
    np.savetxt( path , run_rewards, delimiter=",")

    # return average results
    return [r_avg, np.nan, t_avg, r_max, r_min, r_std_dev]


if __name__ == "__main__":
    #Create base env
    if RENDER:
        base_env = gym.make('CarRacing-v2', render_mode="human").env
    else:
        base_env = gym.make('CarRacing-v2').env

    # Wrap the base environment with dynamic noise effects
    env = RandomizedCarRacingAnnealing(base_env)
    env_test = RandomizedCarRacingTesting(base_env)
    if not TEST:
        # Train Agent
        agent = DDQN_Agent()
        train_agent( agent, env, episodes = EPISODES )
    
    else:
        # Test Agent
        agent = DDQN_Agent()
        test_agent( agent, env_test, model = PRETRAINED_PATH, testnum=10 )
