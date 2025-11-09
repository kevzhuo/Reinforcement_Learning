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


class ActorCriticNetwork(nn.Module):
    """Neural network with separate actor and critic heads."""
    
    def __init__(self, state_size, action_size, hidden_size=128):
        super(ActorCriticNetwork, self).__init__()
        # Shared layers
        self.shared_layer1 = nn.Linear(state_size, hidden_size)
        self.shared_layer2 = nn.Linear(hidden_size, hidden_size)
        
        # Actor head (policy)
        self.actor_head = nn.Linear(hidden_size, action_size)
        
        # Critic head (value function)
        self.critic_head = nn.Linear(hidden_size, 1)

    def forward(self, state):
        """Forward pass returns both policy and value."""
        x = torch.relu(self.shared_layer1(state))
        x = torch.relu(self.shared_layer2(x))
        
        # Actor output: probability distribution over actions
        policy = torch.softmax(self.actor_head(x), dim=-1)
        
        # Critic output: state value
        value = self.critic_head(x)
        
        return policy, value


class ActorCritic:
    """One-step Actor-Critic algorithm implementation."""
    
    def __init__(self, env, actor_lr=0.001, critic_lr=0.001, gamma=0.99, hidden_size=128):
        self.env = env
        self.gamma = gamma
        
        # Get state and action dimensions
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        
        # Initialize actor-critic network
        self.network = ActorCriticNetwork(self.state_size, self.action_size, hidden_size)
        
        # Separate optimizers for actor and critic (optional, can use single optimizer)
        self.actor_optimizer = optim.Adam(
            list(self.network.shared_layer1.parameters()) + 
            list(self.network.shared_layer2.parameters()) + 
            list(self.network.actor_head.parameters()), 
            lr=actor_lr
        )
        self.critic_optimizer = optim.Adam(
            list(self.network.shared_layer1.parameters()) + 
            list(self.network.shared_layer2.parameters()) + 
            list(self.network.critic_head.parameters()), 
            lr=critic_lr
        )
        
    def select_action(self, state):
        """Select action according to current policy."""
        state = torch.FloatTensor(state).unsqueeze(0)
        policy, value = self.network(state)
        m = Categorical(policy)
        action = m.sample()
        return action.item(), m.log_prob(action), value
    
    def train_episode(self):
        """Train for one episode using one-step actor-critic."""
        state, _ = self.env.reset()
        episode_reward = 0
        discount_factor = 1.0
        done = False
        
        while not done:
            # Select action using current policy
            action, log_prob, value = self.select_action(state)
            
            # Take action in environment
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            episode_reward += reward
            
            # Get value of next state (0 if terminal)
            if done:
                next_value = torch.tensor([0.0])
            else:
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
                _, next_value = self.network(next_state_tensor)
                next_value = next_value.detach()  # Don't backprop through next value
            
            # Compute TD error
            td_target = reward + self.gamma * next_value
            td_error = td_target - value
            
            # Compute losses
            critic_loss = 0.5 * td_error.pow(2)
            actor_loss = -td_error.detach() * log_prob
            
            # Zero gradients
            self.critic_optimizer.zero_grad()
            self.actor_optimizer.zero_grad()
            
            # Backward passes
            critic_loss.backward(retain_graph=True)
            actor_loss.backward()
            
            # Update parameters
            self.critic_optimizer.step()
            self.actor_optimizer.step()
            
            state = next_state
        
        return episode_reward
    
    def train(self, num_episodes=1000, print_every=100):
        """Train the actor-critic agent."""
        scores = []
        
        for _ in tqdm.tqdm(range(1, num_episodes + 1)):
            score = self.train_episode()
            scores.append(score)
        
        return scores
    
    def test(self, num_episodes=10, render=False):
        """Test the trained agent."""
        test_scores = []
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # Select action greedily (no exploration)
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                policy, _ = self.network(state_tensor)
                action = torch.argmax(policy).item()
                
                state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
            
            test_scores.append(episode_reward)
            print(f"Test Episode {episode + 1}: Score = {episode_reward:.2f}")
        
        print(f"\nAverage Test Score: {np.mean(test_scores):.2f}")
        return test_scores
    
    def save(self, filepath):
        """Save the model."""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load the model."""
        checkpoint = torch.load(filepath)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        print(f"Model loaded from {filepath}")
    
    def record_video(self, env_name, video_folder='videos', video_prefix='actor-critic-agent'):
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
                policy, _ = self.network(state_tensor)
            action = torch.argmax(policy).item()
            
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        env.close()
        print(f'Video recorded: Total Reward = {total_reward:.2f}, Steps = {steps}')
        print(f'Video saved to {video_folder}/')
        return total_reward, steps

def plot_scores(scores, title="Training Progress"):
    """Plot training scores."""
    plt.figure(figsize=(10, 5))
    plt.plot(scores, alpha=0.6, label='Episode Score')
    
    # Moving average
    window_size = 100
    if len(scores) >= window_size:
        moving_avg = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(scores)), moving_avg, 'r-', linewidth=2, label=f'{window_size}-Episode Moving Avg')
    
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('Policy_Gradient_Methods/actor_critic_training.png')
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='One-Step Actor-Critic')
    parser.add_argument('--env', type=str, default='CartPole-v1', help='Environment name')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--actor_lr', type=float, default=0.001, help='Actor learning rate')
    parser.add_argument('--critic_lr', type=float, default=0.001, help='Critic learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden layer size')
    parser.add_argument('--test', action='store_true', help='Test mode')
    parser.add_argument('--load', type=str, default=None, help='Load model from file')
    parser.add_argument('--save', type=str, default='Policy_Gradient_Methods/actor_critic_model.pth', help='Save model to file')
    parser.add_argument('--record', action='store_true',
                        help='Record video of trained agent after training')
    parser.add_argument('--video-folder', type=str, default='Policy_Gradient_Methods',
                        help='Folder to save videos (default: Policy_Gradient_Methods)')
    parser.add_argument('--video-prefix', type=str, default='actor-critic-agent',
                        help='Prefix for video filenames (default: actor-critic-agent)')
    
    args = parser.parse_args()
    
    # Create environment
    env = gym.make(args.env)
    
    # Create agent
    agent = ActorCritic(
        env, 
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        gamma=args.gamma,
        hidden_size=args.hidden_size
    )
    
    # Load model if specified
    if args.load:
        agent.load(args.load)
    elif args.test:
        # Test mode
        test_env = gym.make(args.env, render_mode='human')
        agent.env = test_env
        agent.test(num_episodes=10, render=True)
    else:
        # Training mode
        print(f"Training Actor-Critic on {args.env}...")
        scores = agent.train(num_episodes=args.episodes)
        
        # Save model
        agent.save(args.save)
        
        # Plot results
        plot_scores(scores, f"Actor-Critic on {args.env}")
        
        # Test the trained agent
        print("\nTesting trained agent...")
        agent.test(num_episodes=5)
        
        # Record video if requested
        if args.record:
            print('\nRecording video of trained agent...')
            env.close()
            agent.record_video(args.env, video_folder=args.video_folder, video_prefix=args.video_prefix)
    
    env.close()


if __name__ == "__main__":
    main()