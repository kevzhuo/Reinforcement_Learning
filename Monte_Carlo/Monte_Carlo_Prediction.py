import gymnasium as gym
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict

def policy(player_sum):
    """
    Simple policy for Blackjack:
    Stick if player_sum > 15, else hit.
    """
    if player_sum > 15:
        return 0
    else:
        return 1


def generate_episode(env):
    """
    Generate a single episode following the defined policy.
    """
    episode = []
    state, _ = env.reset()
    
    while True:
        action = policy(state[0])
        next_state, reward, terminated, truncated, _ = env.step(action)
        episode.append((state, action, reward))
        
        if terminated or truncated:
            break
        state = next_state
    
    return episode


def monte_carlo_prediction(num_episodes=500000, gamma=1.0):
    """
    First-visit Monte Carlo prediction algorithm for Blackjack.
    """
    env = gym.make("Blackjack-v1")
    
    # Initialize return tracking
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)
    V = defaultdict(float)
    
    print(f"Running Monte Carlo prediction for {num_episodes} episodes...")

    for _ in tqdm.tqdm(range(num_episodes)):
        # Generate episode
        episode = generate_episode(env)
        
        # Track states visited in this episode (for first-visit MC)
        visited_states = set()
        
        # Calculate returns for each state (working backwards)
        G = 0
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = gamma * G + reward
            
            # Only update if this is the first visit to this state
            if state not in visited_states:
                visited_states.add(state)
                returns_sum[state] += G
                returns_count[state] += 1
                V[state] = returns_sum[state] / returns_count[state]
    
    env.close()
    print(f"Monte Carlo prediction completed!")
    print(f"Number of unique states visited: {len(V)}")
    
    return V


def plot_value_function_heatmap(V):
    """
    Visualize the value function as heatmaps.
    Separate plots for usable ace and no usable ace.
    """
    # Prepare data for plotting
    player_range = range(12, 22)
    dealer_range = range(1, 11)
    
    # Create grids for plotting
    usable_ace_values = np.zeros((len(player_range), len(dealer_range)))
    no_usable_ace_values = np.zeros((len(player_range), len(dealer_range)))
    
    # Fill the grids with values from V
    for i, player in enumerate(player_range):
        for j, dealer in enumerate(dealer_range):
            state_usable = (player, dealer, True)
            state_no_usable = (player, dealer, False)
            usable_ace_values[i, j] = V.get(state_usable, 0)
            no_usable_ace_values[i, j] = V.get(state_no_usable, 0)
    
    # Create heatmaps
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Heatmap for usable ace
    im1 = ax1.imshow(usable_ace_values, cmap='RdYlGn', aspect='auto', origin='lower')
    ax1.set_xticks(range(len(dealer_range)))
    ax1.set_yticks(range(len(player_range)))
    ax1.set_xticklabels(dealer_range)
    ax1.set_yticklabels(player_range)
    ax1.set_xlabel('Dealer Showing')
    ax1.set_ylabel('Player Sum')
    ax1.set_title('Value Function - Usable Ace')
    fig.colorbar(im1, ax=ax1)
    
    # Heatmap for no usable ace
    im2 = ax2.imshow(no_usable_ace_values, cmap='RdYlGn', aspect='auto', origin='lower')
    ax2.set_xticks(range(len(dealer_range)))
    ax2.set_yticks(range(len(player_range)))
    ax2.set_xticklabels(dealer_range)
    ax2.set_yticklabels(player_range)
    ax2.set_xlabel('Dealer Showing')
    ax2.set_ylabel('Player Sum')
    ax2.set_title('Value Function - No Usable Ace')
    fig.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig('Monte_Carlo/monte_carlo_value_function_heatmap.png', dpi=300, bbox_inches='tight')
    print("Value function heatmap saved as 'monte_carlo_value_function_heatmap.png'")
    plt.show()


if __name__ == "__main__":
    print("Monte Carlo Prediction for Blackjack")
    print("Policy: Stick over 15, Hit otherwise")
    print("="*50)
    
    # Run the Monte Carlo prediction algorithm
    V = monte_carlo_prediction(num_episodes=500000)
    
    # Visualize the value function
    print("\nGenerating visualizations...")
    plot_value_function_heatmap(V)