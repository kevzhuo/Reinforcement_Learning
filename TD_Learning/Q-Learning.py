import gymnasium as gym
import numpy as np
import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
from gymnasium.wrappers import RecordVideo

class QLearning:
    def __init__(self, alpha = 0.1, gamma = 1.0, epsilon = 0.1):
        """
        Q-Learning
        """ 
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
        Generate an episode and update Q-function every step using Q-Learning
        """
        episode = []
        state, _ = env.reset()
        
        while True:
            action = self.behavior_policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, action, reward))
            
            # Q-Learning update
            q_sa = self.Q[state][action]
            if not (terminated or truncated):
                # Use max Q-value over all actions in next state (off-policy)
                max_q_next = max([self.Q[next_state][a] for a in range(4)])
            else:
                max_q_next = 0.0
            
            td_target = reward + self.gamma * max_q_next
            td_error = td_target - q_sa
            self.Q[state][action] += self.alpha * td_error
            
            if terminated or truncated:
                break
            state = next_state
    
    def get_optimal_policy(self):
        """
        Derive the optimal policy from the learned Q-function
        """
        policy = {}
        for state in self.Q:
            q_values = [self.Q[state][a] for a in range(4)]
            best_action = np.argmax(q_values)
            policy[state] = best_action
        return policy
    
    def train(self, env, num_episodes=100000):
        """
        Train the Q-Learning agent
        """
        env = gym.make('CliffWalking-v1')

        print(f"Training Q-Learning for {num_episodes} episodes...")
        
        for _ in tqdm.tqdm(range(num_episodes)):
            # Generate episode and update Q-function
            self.generate_episode_and_update(env)
        
        env.close()
        
        return self.get_optimal_policy()
    
    def evaluate_policy(self, env, policy, video_folder="videos", video_prefix="qlearning-agent"):
        """
        Evaluate the policy by showing the agent's path and recording a video
        """
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
    qlearning_agent = QLearning(alpha=0.5, gamma=1.0, epsilon=0.1)
    env = gym.make("CliffWalking-v1", render_mode="rgb_array")
    optimal_policy = qlearning_agent.train(env, num_episodes=200000)

    # Final evaluation with video recording
    print("\nEvaluating the learned policy:")
    env = gym.make("CliffWalking-v1", render_mode="rgb_array")
    final_reward, final_steps = qlearning_agent.evaluate_policy(env, optimal_policy, video_folder="TD_Learning", video_prefix="qlearning-final-policy")
    print(f"Final evaluation: {final_steps} steps, total reward = {final_reward}")