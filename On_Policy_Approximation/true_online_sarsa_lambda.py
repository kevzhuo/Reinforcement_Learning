import numpy as np
import gymnasium as gym
import tqdm
from gymnasium.wrappers import RecordVideo
from tile_coder import TileCoder

class TrueOnlineSarsaLambda:
    def __init__(self, env, tile_coder, epsilon=0.1, gamma=1.0, lambda_=0.9, num_actions=3):
        self.env = env
        self.tile_coder = tile_coder
        self.alpha = 0.5 / tile_coder.num_tilings
        self.epsilon = epsilon
        self.gamma = gamma
        self.lambda_ = lambda_
        self.num_actions = num_actions

        # Initialize weights for each action
        self.weights = [np.zeros(tile_coder.total_num_tiles) for _ in range(num_actions)]

    def get_features(self, state):
        active_tiles = self.tile_coder.get_tiles(state)
        return active_tiles
    
    def get_Q_value(self, state, action):
        active_tiles = self.get_features(state)
        return np.sum(self.weights[action][active_tiles])
    
    def epsilon_greedy_policy(self, state):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.num_actions)
        else:
            q_values = [self.get_Q_value(state, a) for a in range(self.num_actions)]
            return np.argmax(q_values)

    def generate_episode_and_update(self):
        # Initialize eligibility trace (one for each action's weight vector)
        z = [np.zeros(self.tile_coder.total_num_tiles) for _ in range(self.num_actions)]
        
        state, _ = self.env.reset()
        action = self.epsilon_greedy_policy(state)
        
        # Get initial features and Q-value
        active_tiles = self.get_features(state)
        q_old = 0.0  
        
        total_reward = 0
        
        while True:
            # Get current Q-value
            q_current = self.get_Q_value(state, action)
            
            # Take action A, observe R and S'
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            total_reward += reward
            done = terminated or truncated
            
            # Compute TD error (delta)
            delta = reward - q_current
            
            # Update eligibility traces with Dutch trace for current action only
            # First decay all traces
            for a in range(self.num_actions):
                z[a] *= self.gamma * self.lambda_
            
            # For the current action, add the feature vector with Dutch trace correction
            # z^T * x is the sum of trace values at active tiles for this action
            z_dot_x = np.sum(z[action][active_tiles])
            
            # Add the feature vector with correction term
            # For binary features, x_i = 1 for active tiles, 0 elsewhere
            z[action][active_tiles] += (1.0 - self.alpha * self.gamma * self.lambda_ * z_dot_x)
            
            if done:
                # Terminal state: update weights
                # w <- w + alpha * [delta + Q(S,A) - Q_old] * z - alpha * [Q(S,A) - Q_old] * x
                for a in range(self.num_actions):
                    self.weights[a] += self.alpha * (delta + q_current - q_old) * z[a]
                
                # Subtract the correction term for active features of current action
                self.weights[action][active_tiles] -= self.alpha * (q_current - q_old)
                
                break
            
            # Non-terminal state: choose A' from S' using policy
            next_action = self.epsilon_greedy_policy(next_state)
            
            # Get next features and Q-value
            next_active_tiles = self.get_features(next_state)
            q_next = self.get_Q_value(next_state, next_action)
            
            # Complete TD error calculation
            delta += self.gamma * q_next
            
            # Update weights
            # w <- w + alpha * [delta + Q(S,A) - Q_old] * z - alpha * [Q(S,A) - Q_old] * x
            for a in range(self.num_actions):
                self.weights[a] += self.alpha * (delta + q_current - q_old) * z[a]
            
            # Subtract the correction term for active features of current action
            self.weights[action][active_tiles] -= self.alpha * (q_current - q_old)
            
            # Update for next iteration
            q_old = q_next
            state = next_state
            action = next_action
            active_tiles = next_active_tiles
        
        return total_reward

    def train(self, num_episodes=500):
        for _ in tqdm.tqdm(range(num_episodes)):
            self.generate_episode_and_update()

        self.env.close()
    
    def evaluate_policy(self, video_folder="videos", video_prefix="true-online-sarsa-lambda-agent"):
        env = RecordVideo(
            self.env,
            video_folder=video_folder,
            name_prefix=video_prefix,
        )
        
        state, _ = env.reset()
        total_steps = 0
        total_reward = 0
        
        while True:
            action = np.argmax([self.get_Q_value(state, a) for a in range(self.num_actions)])
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_steps += 1
            total_reward += reward
            if terminated or truncated:
                break
            state = next_state
        
        env.close()
        
        print(f"Video saved to: {video_folder}/")
        
        return total_steps, total_reward
    
if __name__ == "__main__":
    env = gym.make("MountainCar-v0", render_mode="rgb_array")
    tile_coder = TileCoder(num_tiles=8, num_tilings=8, state_bounds=[[-1.2, 0.6], [-0.07, 0.07]])
    agent = TrueOnlineSarsaLambda(env, tile_coder, lambda_=0.9)
    agent.train(num_episodes=50000)

    # Evaluate the learned policy
    print("\nEvaluating the learned policy:")
    final_steps, final_reward = agent.evaluate_policy(video_folder="On_Policy_Approximation", video_prefix="true-online-sarsa-lambda-final-policy")
    print(f"Final evaluation: {final_steps} steps, total reward = {final_reward}")
