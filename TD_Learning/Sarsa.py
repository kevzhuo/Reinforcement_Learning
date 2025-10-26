import gymnasium as gym
import numpy as np
import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
from gymnasium.wrappers import RecordVideo

class Sarsa:
    def __init__(self, alpha = 0.1, gamma = 1.0, epsilon = 0.1):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # Q-function: Q(s,a) -> expected return
        self.Q = defaultdict(lambda: defaultdict(float))
        
    def behavior_policy(self, state):
        """
        Epsilon-greedy behavior policy
        """
        if np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.choice([0, 1, 2, 3])
        else:
            # Exploit: greedy action
            q_values = [self.Q[state][0], self.Q[state][1], self.Q[state][2], self.Q[state][3]]
            return np.argmax(q_values)
    
    def generate_episode_and_update(self, env):
        """
        Generate an episode and update Q-function every step using Sarsa
        """
        episode = []
        state, _ = env.reset()
        
        while True:
            action = self.behavior_policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, action, reward))
            
            # Choose next action using behavior policy
            if terminated or truncated:
                next_action = None
            else:
                next_action = self.behavior_policy(next_state)
            
            # Sarsa update
            q_sa = self.Q[state][action]
            if next_action is not None:
                q_snext_anext = self.Q[next_state][next_action]
            else:
                q_snext_anext = 0.0
            
            td_target = reward + self.gamma * q_snext_anext
            td_error = td_target - q_sa
            self.Q[state][action] += self.alpha * td_error
            
            if terminated or truncated:
                break
            state = next_state
    
    def get_optimal_policy(self):
        """
        Extract the optimal policy from the Q-function
        """
        policy = {}
        for state in self.Q:
            q_values = [self.Q[state][0], self.Q[state][1], self.Q[state][2], self.Q[state][3]]
            policy[state] = np.argmax(q_values)
        return policy
    
    def train(self, env, num_episodes=100000):
        """
        Train the Sarsa agent
        """
        env = gym.make('CliffWalking-v1')

        print(f"Training Sarsa for {num_episodes} episodes...")
        
        for _ in tqdm.tqdm(range(num_episodes)):
            # Generate episode and update Q-function
            self.generate_episode_and_update(env)
        
        env.close()
        
        return self.get_optimal_policy()

    def evaluate_policy(self, env, policy, video_folder="videos", video_prefix="sarsa-agent"):
        # Wrap environment with RecordVideo to capture the episode
        env = RecordVideo(
            env,
            video_folder=video_folder,
            name_prefix=video_prefix,
        )
        
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            action = policy.get(state, 0)
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
            state = next_state
        
        env.close()
        
        print(f"Episode completed: {steps} steps, total reward = {total_reward}")
        print(f"Video saved to: {video_folder}/")
        
        return total_reward, steps

if __name__ == "__main__":
    sarsa_agent = Sarsa(alpha=0.5, gamma=1.0, epsilon=0.1)
    env = gym.make("CliffWalking-v1", render_mode="rgb_array")
    optimal_policy = sarsa_agent.train(env, num_episodes=200000)

    # Final evaluation with video recording
    print("\nEvaluating the learned policy:")
    env = gym.make("CliffWalking-v1", render_mode="rgb_array")
    final_reward, final_steps = sarsa_agent.evaluate_policy(env, optimal_policy, video_folder="TD_Learning", video_prefix="sarsa-final-policy")
    print(f"Final evaluation: {final_steps} steps, total reward = {final_reward}")