"""
REINFORCE Algorithm (Monte Carlo Policy Gradient)
This implementation supports OpenAI Gymnasium environments via CLI.
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.distributions import Categorical
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from collections import deque
import matplotlib.pyplot as plt


class PolicyNetwork(nn.Module):
    """Neural network for policy approximation."""
    
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super(PolicyNetwork, self).__init__()
        self.layer1 = nn.Linear(state_dim, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, action_dim)
        
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return torch.softmax(x, dim=-1)


class REINFORCE:
    """REINFORCE algorithm implementation."""
    
    def __init__(self, env, learning_rate=0.001, gamma=0.99, hidden_size=128):
        self.env = env
        self.gamma = gamma
        
        # Get state and action dimensions
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        
        # Initialize policy network
        self.policy = PolicyNetwork(self.state_dim, self.action_dim, hidden_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # Episode storage
        self.saved_log_probs = []
        self.rewards = []
        
    def select_action(self, state):
        """Select action according to current policy."""
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy(state)
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        return action.item()
    
    def compute_returns(self):
        """Compute discounted returns for each timestep."""
        returns = []
        G = 0
        
        # Compute returns backwards
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        
        # Normalize returns for stability
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        return returns
    
    def update_policy(self):
        """Update policy using REINFORCE algorithm."""
        returns = self.compute_returns()
        
        policy_loss = []
        for log_prob, G in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * G)
        
        # Compute total loss and perform backprop
        self.optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        
        # Clear episode data
        self.saved_log_probs = []
        self.rewards = []
        
        return policy_loss.item()
    
    def train(self, num_episodes=1000, render=False):
        """Train the REINFORCE agent."""
        scores = []
        
        for _ in tqdm.tqdm(range(1, num_episodes + 1)):
            state, _ = self.env.reset()
            episode_reward = 0
            
            # Run episode
            while True:
                action = self.select_action(state)
                state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                self.rewards.append(reward)
                episode_reward += reward
                
                if render:
                    self.env.render()
                
                if done:
                    break
            
            # Update policy
            self.update_policy()
            
            # Track scores
            scores.append(episode_reward)
        
        return scores
    
    def test(self, num_episodes=10, render=True):
        """Test the trained agent."""
        test_rewards = []
        
        for _ in tqdm.tqdm(range(num_episodes)):
            state, _ = self.env.reset()
            episode_reward = 0
            
            while True:
                # Use greedy action selection (no exploration)
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    probs = self.policy(state_tensor)
                action = torch.argmax(probs).item()
                
                state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
                
                if render:
                    self.env.render()
                
                if done:
                    break
            
            test_rewards.append(episode_reward)
        
        print(f'\nAverage Test Reward: {np.mean(test_rewards):.2f}')
        return test_rewards
    
    def save_model(self, filepath='Policy_Gradient_Methods/reinforce_model.pth'):
        """Save the trained policy network."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
        print(f'Model saved to {filepath}')
    
    def load_model(self, filepath='Policy_Gradient_Methods/reinforce_model.pth'):
        """Load a trained policy network."""
        checkpoint = torch.load(filepath)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f'Model loaded from {filepath}')
    
    def record_video(self, env_name, video_folder='videos', video_prefix='reinforce-agent'):
        """Record a video of the trained agent."""
        # Create a new environment with video recording
        env = gym.make(env_name, render_mode='rgb_array')
        env = RecordVideo(
            env,
            video_folder=video_folder,
            name_prefix=video_prefix,
        )
        
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            # Use greedy action selection
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                probs = self.policy(state_tensor)
            action = torch.argmax(probs).item()
            
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        env.close()
        print(f'Video recorded: Total Reward = {total_reward:.2f}, Steps = {steps}')
        print(f'Video saved to {video_folder}/')
        return total_reward, steps


def plot_scores(scores, title='REINFORCE Training Progress'):
    """Plot training scores."""
    plt.figure(figsize=(10, 5))
    plt.plot(scores)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title(title)
    
    # Add moving average
    window_size = 100
    if len(scores) >= window_size:
        moving_avg = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(scores)), moving_avg, 'r-', linewidth=2, label='Moving Average')
        plt.legend()
    
    plt.grid(True)
    plt.savefig('Policy_Gradient_Methods/reinforce_training.png')
    print('Training plot saved')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='REINFORCE Algorithm for OpenAI Gymnasium')
    parser.add_argument('--env', type=str, default='CartPole-v1',
                        help='Gymnasium environment name (default: CartPole-v1)')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of training episodes (default: 1000)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor (default: 0.99)')
    parser.add_argument('--hidden-size', type=int, default=128,
                        help='Hidden layer size (default: 128)')
    parser.add_argument('--test', action='store_true',
                        help='Test mode (load and test trained model)')
    parser.add_argument('--model-path', type=str, default='Policy_Gradient_Methods/reinforce_model.pth',
                        help='Path to save/load model (default: reinforce_model.pth)')
    parser.add_argument('--render', action='store_true',
                        help='Render environment during training/testing')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (default: None)')
    parser.add_argument('--record-video', action='store_true',
                        help='Record video of trained agent after training')
    parser.add_argument('--video-folder', type=str, default='Policy_Gradient_Methods',
                        help='Folder to save videos (default: videos)')
    parser.add_argument('--video-prefix', type=str, default='reinforce-agent',
                        help='Prefix for video filenames (default: reinforce-agent)')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    
    # Create environment
    print(f'Creating environment: {args.env}')
    env = gym.make(args.env, render_mode='human' if args.render else None)
    
    # Initialize REINFORCE agent
    agent = REINFORCE(env, learning_rate=args.lr, gamma=args.gamma, hidden_size=args.hidden_size)
    
    if args.test:
        # Test mode
        print('Loading trained model...')
        agent.load_model(args.model_path)
        print('Testing agent...')
        agent.test(num_episodes=10, render=True)
    else:
        # Training mode
        print('Starting training...')
        print(f'Environment: {args.env}')
        print(f'Episodes: {args.episodes}')
        print(f'Learning Rate: {args.lr}')
        print(f'Gamma: {args.gamma}')
        print(f'Hidden Size: {args.hidden_size}')
        print('-' * 50)
        
        scores = agent.train(num_episodes=args.episodes, render=args.render)
        
        # Save model
        agent.save_model(args.model_path)
        
        # Plot results
        plot_scores(scores, title=f'REINFORCE on {args.env}')
        
        # Record video if requested
        if args.record_video:
            print('\nRecording video of trained agent...')
            env.close()
            agent.record_video(args.env, video_folder=args.video_folder, video_prefix=args.video_prefix)
        else:
            env.close()
    
    # Close environment if in test mode
    if args.test:
        env.close()


if __name__ == '__main__':
    main()
