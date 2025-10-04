import gymnasium as gym
import numpy as np
import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt


class OnPolicyMC:
    def __init__(self, epsilon=0.1, gamma=1.0):
        """
        Monte Carlo On-Policy Control for Blackjack
        """
        self.epsilon = epsilon  # Exploration probability
        self.gamma = gamma  # Discount factor

        # Q-function: Q(s,a) -> expected return
        self.Q = defaultdict(lambda: defaultdict(float))

        # Cumulative weights for weighted importance sampling
        self.returns_sum = defaultdict(lambda: defaultdict(float))
        self.returns_count = defaultdict(lambda: defaultdict(int))

        # Performance tracking
        self.episode_rewards = []
        self.episode_lengths = []

    def behavior_policy(self, state):
        """
        Epsilon-greedy behavior policy
        """
        if np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.choice([0, 1])
        else:
            # Exploit: greedy action
            q_values = [self.Q[state][0], self.Q[state][1]]
            return np.argmax(q_values)

    def generate_episode(self, env):
        """
        Generate an episode using the behavior policy
        """
        episode = []
        state, _ = env.reset()

        while True:
            action = self.behavior_policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, action, reward))

            if terminated or truncated:
                break
            state = next_state

        return episode

    def update_q_function(self, episode):
        """
        Update Q-function using first-visit Monte Carlo method
        """
        # Track visited state-action pairs for first-visit MC
        visited_state_actions = set()

        # Calculate returns working backwards
        G = 0
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = self.gamma * G + reward

            # Only update if this is the first visit to this state-action pair
            state_action = (state, action)
            if state_action not in visited_state_actions:
                visited_state_actions.add(state_action)

                # Update return tracking
                self.returns_sum[state][action] += G
                self.returns_count[state][action] += 1

                # Update Q-function
                self.Q[state][action] = (
                    self.returns_sum[state][action] / self.returns_count[state][action]
                )

    def get_optimal_policy(self):
        """
        Extract the optimal policy from the Q-function
        """
        policy = {}
        for state in self.Q:
            q_values = [self.Q[state][0], self.Q[state][1]]
            policy[state] = np.argmax(q_values)
        return policy

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
                # Use deterministic policy (no exploration)
                if state in policy:
                    action = policy[state]
                else:
                    # Default action for unseen states
                    action = 0 if state[0] >= 20 else 1

                state, reward, terminated, truncated, _ = env.step(action)
                episode_reward = reward  # In Blackjack, only final reward matters

                if terminated or truncated:
                    break

            if episode_reward > 0:
                wins += 1
            total_reward += episode_reward

        return wins / num_episodes, total_reward / num_episodes

    def train(self, env, num_episodes=100000):
        """
        Train the Monte Carlo control algorithm
        """
        print(f"Training Monte Carlo Control for {num_episodes} episodes...")

        for _ in tqdm.tqdm(range(num_episodes)):
            # Generate episode and update Q-function
            episode = self.generate_episode(env)
            self.update_q_function(episode)

        env.close()
        print(f"\nTraining completed!")
        print(f"Total states explored: {len(self.Q)}")

        return self.get_optimal_policy()


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
    ax1.set_title("Optimal Policy - Usable Ace")
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
    ax2.set_title("Optimal Policy - No Usable Ace")
    cbar2 = fig.colorbar(im2, ax=ax2, ticks=[0, 1])
    cbar2.ax.set_yticklabels(["Stick", "Hit"])

    plt.tight_layout()
    plt.savefig(
        "Monte_Carlo/on_policy_mc_policy_heatmap.png",
        dpi=300,
        bbox_inches="tight",
    )
    print("Policy heatmap saved as 'on_policy_mc_policy_heatmap.png'")
    plt.show()


if __name__ == "__main__":
    env = gym.make("Blackjack-v1")
    on_policy_mc = OnPolicyMC(epsilon=1.0, gamma=1.0)
    optimal_policy = on_policy_mc.train(env, num_episodes=200000)

    # Final evaluation
    win_rate, avg_reward = on_policy_mc.evaluate_policy(env, optimal_policy, 10000)
    print(f"\nFinal Policy Evaluation over 10,000 episodes:")
    print(f"  Win rate: {win_rate:.4f}")
    print(f"  Average reward: {avg_reward:.4f}")
    env.close()

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_policy_heatmap(optimal_policy)
