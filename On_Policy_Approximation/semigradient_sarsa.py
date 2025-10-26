import numpy as np
import gymnasium as gym
import tqdm
from gymnasium.wrappers import RecordVideo
from tile_coder import TileCoder

class SemiGradientSarsa:
    def __init__(self, env, tile_coder, epsilon=0.1, gamma=1.0, num_actions=3):
        self.env = env
        self.tile_coder = tile_coder
        self.alpha = 0.5 / tile_coder.num_tilings
        self.epsilon = epsilon
        self.gamma = gamma
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

        state, _ = self.env.reset()
        action = self.epsilon_greedy_policy(state)
        total_reward = 0
        
        while True:
            # Take action A, observe R and S'
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            total_reward += reward
            done = terminated or truncated
            
            # Get active tiles for current state-action pair
            active_tiles = self.get_features(state)
            
            if done:
                # Terminal state: TD target is just the reward
                # Update: w <- w + alpha * [R - Q(S,A)] * grad_Q(S,A)
                q_current = self.get_Q_value(state, action)
                td_error = reward - q_current
                
                # Update weights for the action taken
                self.weights[action][active_tiles] += self.alpha * td_error
                break
            
            # Non-terminal state: choose A' from S' using policy
            next_action = self.epsilon_greedy_policy(next_state)
            
            # Compute TD target: R + gamma * Q(S', A')
            q_current = self.get_Q_value(state, action)
            q_next = self.get_Q_value(next_state, next_action)
            td_target = reward + self.gamma * q_next
            td_error = td_target - q_current
            
            # Update weights: w <- w + alpha * [R + gamma*Q(S',A') - Q(S,A)] * grad_Q(S,A)
            self.weights[action][active_tiles] += self.alpha * td_error
            
            state = next_state
            action = next_action
        
        return total_reward

    def train(self, num_episodes=500):
        for _ in tqdm.tqdm(range(num_episodes)):
            self.generate_episode_and_update()

        self.env.close()
    
    def evaluate_policy(self, video_folder="videos", video_prefix="qlearning-agent" ):
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
    agent = SemiGradientSarsa(env, tile_coder)
    agent.train(num_episodes=50000)

    # Evaluate the learned policy
    print("\nEvaluating the learned policy:")
    final_steps, final_reward = agent.evaluate_policy(video_folder="On_Policy_Approximation", video_prefix="semigradient-sarsa-final-policy")
    print(f"Final evaluation: {final_steps} steps, total reward = {final_reward}")