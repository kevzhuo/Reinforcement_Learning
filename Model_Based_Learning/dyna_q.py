import gymnasium as gym
import numpy as np
import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
from gymnasium.wrappers import RecordVideo

class DynaQ:
    def __init__(self, alpha=0.1, gamma=1.0, epsilon=0.1, n_planning_steps=5):
        """
        Dyna-Q: Integrates learning, planning, and acting
        
        Parameters:
        - alpha: learning rate
        - gamma: discount factor
        - epsilon: exploration rate
        - n_planning_steps: number of planning steps per real step
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_planning_steps = n_planning_steps
        
        # Q-function: Q(s,a) -> expected return
        self.Q = defaultdict(lambda: defaultdict(float))
        
        # Model: stores observed transitions
        # Model[state][action] = (next_state, reward)
        self.model = defaultdict(lambda: defaultdict(lambda: None))
        
        # Keep track of all state-action pairs we've observed
        self.observed_state_actions = []
        
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
    
    def update_q(self, state, action, reward, next_state, terminated):
        """
        Q-Learning update rule
        """
        q_sa = self.Q[state][action]
        if not terminated:
            max_q_next = max([self.Q[next_state][a] for a in range(4)])
        else:
            max_q_next = 0.0
        
        td_target = reward + self.gamma * max_q_next
        td_error = td_target - q_sa
        self.Q[state][action] += self.alpha * td_error
    
    def update_model(self, state, action, next_state, reward):
        """
        Update the model with observed transition
        """
        # Store the transition in the model
        if self.model[state][action] is None:
            # This is a new state-action pair
            self.observed_state_actions.append((state, action))
        
        self.model[state][action] = (next_state, reward)
    
    def planning(self):
        """
        Planning: sample from model and update Q-function
        """
        for _ in range(self.n_planning_steps):
            # Randomly sample a previously observed state-action pair
            if len(self.observed_state_actions) == 0:
                break
                
            state, action = self.observed_state_actions[
                np.random.randint(len(self.observed_state_actions))
            ]
            
            # Get simulated next state and reward from model
            next_state, reward = self.model[state][action]
            
            # Determine if this was a terminal state
            # In the model, we assume the episode terminates if reward is significantly negative
            # This is specific to CliffWalking where falling off cliff gives -100
            terminated = (reward == -100)
            
            # Update Q-function with simulated experience
            self.update_q(state, action, reward, next_state, terminated)
    
    def generate_episode_and_update(self, env):
        """
        Generate an episode and update Q-function and model at every step
        """
        episode = []
        state, _ = env.reset()
        
        while True:
            # (a) Take action according to behavior policy
            action = self.behavior_policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, action, reward))
            
            # (b) Direct RL: Update Q-function from real experience
            self.update_q(state, action, reward, next_state, terminated or truncated)
            
            # (c) Model Learning: Update model with observed transition
            self.update_model(state, action, next_state, reward)
            
            # (d) Planning: Update Q-function using simulated experiences from model
            self.planning()
            
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
        Train the Dyna-Q agent
        """
        env = gym.make('CliffWalking-v1')
        
        print(f"Training Dyna-Q for {num_episodes} episodes with {self.n_planning_steps} planning steps...")
        
        for _ in tqdm.tqdm(range(num_episodes)):
            # Generate episode and update Q-function and model
            self.generate_episode_and_update(env)
        
        env.close()
        
        return self.get_optimal_policy()
    
    def evaluate_policy(self, env, policy, video_folder="videos", video_prefix="dynaq-agent"):
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
    # Try different planning step values to see the impact
    # Higher n_planning_steps should lead to faster learning
    dynaq_agent = DynaQ(alpha=0.5, gamma=1.0, epsilon=0.1, n_planning_steps=10)
    env = gym.make("CliffWalking-v1", render_mode="rgb_array")
    optimal_policy = dynaq_agent.train(env, num_episodes=200000)
    
    # Final evaluation with video recording
    print("\nEvaluating the learned policy:")
    env = gym.make("CliffWalking-v1", render_mode="rgb_array")
    final_reward, final_steps = dynaq_agent.evaluate_policy(env, optimal_policy, video_folder="Model_Based_Learning", video_prefix="dynaq-final-policy")
    print(f"Final evaluation: {final_steps} steps, total reward = {final_reward}")

