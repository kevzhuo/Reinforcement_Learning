import gymnasium as gym
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import tqdm

class OffPolicyMonteCarloControl:
    def __init__(self, epsilon=0.1, gamma=1.0):
        """
        Monte Carlo Off-Policy Control for Blackjack using Importance Sampling
        """
        self.epsilon = epsilon
        self.gamma = gamma
        
        # Q-function: Q(s,a) -> expected return
        self.Q = defaultdict(lambda: defaultdict(float))
        
        # Cumulative weights for weighted importance sampling
        self.C = defaultdict(lambda: defaultdict(float))

        
    def behavior_policy(self, state):
        """
        Epsilon-greedy behavior policy (used to generate episodes)
        """
        if np.random.random() < self.epsilon:
            # Explore: random action
            action = np.random.choice([0, 1])
            action_prob = self.epsilon / 2.0
        else:
            # Exploit: greedy action based on current Q-function
            q_values = [self.Q[state][0], self.Q[state][1]]
            action = np.argmax(q_values)
            action_prob = 1.0 - self.epsilon + (self.epsilon / 2.0)
        
        return action, action_prob
    
    def target_policy(self, state):
        """
        Greedy target policy (the policy we want to evaluate/improve)
        """
        q_values = [self.Q[state][0], self.Q[state][1]]
        action = np.argmax(q_values)
        return action, 1.0
    
    def get_action_probability(self, state, action, policy_type='behavior'):
        """
        Get probability of taking an action under a specific policy
        """
        if policy_type == 'behavior':
            # Epsilon-greedy probability
            q_values = [self.Q[state][0], self.Q[state][1]]
            greedy_action = np.argmax(q_values)
            
            if action == greedy_action:
                return 1.0 - self.epsilon + (self.epsilon / 2.0)
            else:
                return self.epsilon / 2.0
        
        elif policy_type == 'target':
            # Greedy probability
            q_values = [self.Q[state][0], self.Q[state][1]]
            greedy_action = np.argmax(q_values)
            
            return 1.0 if action == greedy_action else 0.0
    
    def generate_episode(self, env):
        """
        Generate an episode using the behavior policy
        """
        episode = []
        state, _ = env.reset()
        
        while True:
            action, action_prob = self.behavior_policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, action, reward, action_prob))
            
            if terminated or truncated:
                break
            state = next_state
        
        return episode
    
    def off_policy_mc_control(self, episode):
        """
        Off-policy Monte Carlo control using weighted importance sampling
        """
        G = 0.0  # Return
        W = 1.0  # Importance sampling weight
        
        # Process episode backwards (from terminal state)
        for t in reversed(range(len(episode))):
            state, action, reward, behavior_prob = episode[t]
            
            # Update return
            G = self.gamma * G + reward
            
            # Update cumulative weight
            self.C[state][action] += W
            
            # Update Q-function using weighted importance sampling
            if self.C[state][action] > 0:
                self.Q[state][action] += (W / self.C[state][action]) * (G - self.Q[state][action])
            
            # Calculate importance ratio for this step
            target_prob = self.get_action_probability(state, action, 'target')
            
            # If target policy probability is 0, terminate importance sampling
            if target_prob == 0:
                break
            
            # Update importance sampling weight
            W *= target_prob / behavior_prob
    
    def evaluate_policy(self, env, policy, num_episodes=10000):
        """
        Evaluate a policy by running episodes and calculating win rate
        """
        wins = 0
        total_reward = 0
        
        for _ in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            
            while True:
                # Use deterministic target policy
                action, _ = self.target_policy(state)
                state, reward, terminated, truncated, _ = env.step(action)
                episode_reward = reward 
                
                if terminated or truncated:
                    break
            
            if episode_reward > 0:
                wins += 1
            total_reward += episode_reward
        
        return wins / num_episodes, total_reward / num_episodes

    def train(self, env, num_episodes=100000):
        """
        Train using off-policy Monte Carlo control
        """
        print(f"Training Off-Policy Monte Carlo Control for {num_episodes} episodes...")
        print(f"Behavior policy: ε-greedy with ε = {self.epsilon}")
        print(f"Target policy: Greedy")
        
        for _ in tqdm.tqdm(range(num_episodes)):
            # Generate episode using behavior policy
            episode = self.generate_episode(env)
            
            # Update Q-function using off-policy control
            self.off_policy_mc_control(episode)
            
            # Calculate average importance ratio for this episode
            total_weight = 1.0
            for state, action, reward, behavior_prob in episode:
                target_prob = self.get_action_probability(state, action, 'target')
                if target_prob > 0:
                    total_weight *= target_prob / behavior_prob
                else:
                    total_weight = 0
                    break
        
        env.close()
        print(f"\nTraining completed!")
        print(f"Total states explored: {len(self.Q)}")
        
        return self.get_target_policy()

    def get_target_policy(self):
        """
        Extract the target policy from the Q-function
        """
        policy = {}
        for state in self.Q:
            q_values = [self.Q[state][0], self.Q[state][1]]
            policy[state] = np.argmax(q_values)
        return policy


def plot_policy_heatmap(policy):
    """
    Visualize the optimal policy as heatmaps.
    """
    # Prepare data for plotting
    player_range = range(12, 22)
    dealer_range = range(1, 11)

    # Create grids for plotting
    usable_ace_policy = np.zeros((len(player_range), len(dealer_range)))
    no_usable_ace_policy = np.zeros((len(player_range), len(dealer_range)))

    # Fill the grids with policy actions
    for i, player in enumerate(player_range):
        for j, dealer in enumerate(dealer_range):
            state_usable = (player, dealer, True)
            state_no_usable = (player, dealer, False)
            usable_ace_policy[i, j] = policy.get(state_usable, 0)
            no_usable_ace_policy[i, j] = policy.get(state_no_usable, 1)

    # Create heatmaps
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Heatmap for usable ace
    im1 = ax1.imshow(
        usable_ace_policy,
        cmap="RdYlGn_r",
        aspect="auto",
        origin="lower",
        vmin=0,
        vmax=1,
    )
    ax1.set_xticks(range(len(dealer_range)))
    ax1.set_yticks(range(len(player_range)))
    ax1.set_xticklabels(dealer_range)
    ax1.set_yticklabels(player_range)
    ax1.set_xlabel("Dealer Showing")
    ax1.set_ylabel("Player Sum")
    ax1.set_title("Optimal Policy - Usable Ace (Off-Policy MC)")
    cbar1 = fig.colorbar(im1, ax=ax1, ticks=[0, 1])
    cbar1.ax.set_yticklabels(["Stick", "Hit"])

    # Heatmap for no usable ace
    im2 = ax2.imshow(
        no_usable_ace_policy,
        cmap="RdYlGn_r",
        aspect="auto",
        origin="lower",
        vmin=0,
        vmax=1,
    )
    ax2.set_xticks(range(len(dealer_range)))
    ax2.set_yticks(range(len(player_range)))
    ax2.set_xticklabels(dealer_range)
    ax2.set_yticklabels(player_range)
    ax2.set_xlabel("Dealer Showing")
    ax2.set_ylabel("Player Sum")
    ax2.set_title("Optimal Policy - No Usable Ace (Off-Policy MC)")
    cbar2 = fig.colorbar(im2, ax=ax2, ticks=[0, 1])
    cbar2.ax.set_yticklabels(["Stick", "Hit"])

    plt.tight_layout()
    plt.savefig(
        "Monte_Carlo/off_policy_mc_policy_heatmap.png",
        dpi=300,
        bbox_inches="tight",
    )
    print("Policy heatmap saved as 'off_policy_mc_policy_heatmap.png'")
    plt.show()
    

if __name__ == "__main__":  
    env = gym.make("Blackjack-v1")
    off_policy_mc = OffPolicyMonteCarloControl(
        epsilon=0.1,   
        gamma=1.0       
    )
    
    # Train the algorithm
    target_policy = off_policy_mc.train(env, num_episodes=200000)
    
    # Final policy evaluation
    final_win_rate, final_avg_reward = off_policy_mc.evaluate_policy(env, target_policy, 10000)
    env.close()

    print(f"\nFinal Policy Evaluation over 10,000 episodes:")
    print(f"  Win rate: {final_win_rate:.4f}")
    print(f"  Average reward: {final_avg_reward:.4f}")

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_policy_heatmap(target_policy)