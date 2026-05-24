import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

#  Hyperparameters 
EPISODES = 500
ALPHA    = 0.1
GAMMA    = 0.99
WINDOW   = 20

#  Agent class 
class QLearningAgent:
    def __init__(self, n_states, n_actions, eps_start, eps_end, eps_decay, name):
        self.q       = np.zeros((n_states, n_actions))
        self.eps     = eps_start
        self.eps_end = eps_end
        self.decay   = eps_decay
        self.name    = name
        self.n_actions = n_actions

    def act(self, state):
        if np.random.rand() < self.eps:
            return np.random.randint(self.n_actions)
        return np.argmax(self.q[state])

    def learn(self, s, a, r, s2, done):
        target = r + (0 if done else GAMMA * np.max(self.q[s2]))
        self.q[s, a] += ALPHA * (target - self.q[s, a])

    def step_epsilon(self):
        self.eps = max(self.eps_end, self.eps * self.decay)

# Create 3 agents 
env = gym.make("CliffWalking-v1")
S, A = env.observation_space.n, env.action_space.n
env.close()

agents = [
    QLearningAgent(S, A, eps_start=0.50, eps_end=0.50, eps_decay=1.0,   name="ε=0.50 constant"),
    QLearningAgent(S, A, eps_start=0.05, eps_end=0.05, eps_decay=1.0,   name="ε=0.05 constant"),
    QLearningAgent(S, A, eps_start=1.00, eps_end=0.01, eps_decay=0.995, name="ε decaying 1→0.01"),
]

# Train
all_rewards = {}

for agent in agents:
    rewards = []
    env = gym.make("CliffWalking-v1")

    for ep in range(EPISODES):
        obs, _ = env.reset()
        total, done = 0, False

        while not done:
            action = agent.act(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            agent.learn(obs, action, reward, next_obs, terminated)
            obs, done, total = next_obs, terminated or truncated, total + reward

        agent.step_epsilon()
        rewards.append(total)

    env.close()
    all_rewards[agent.name] = rewards
    print(f"Done: {agent.name} | final avg: {np.mean(rewards[-50:]):.1f}")

# Plot
COLORS = ["#E85D4A", "#3A86FF", "#2DC653"]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                                gridspec_kw={"height_ratios": [3, 1]})
fig.patch.set_facecolor("#0F1117")
for ax in (ax1, ax2):
    ax.set_facecolor("#0F1117")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2A2D3A")

for (name, rewards), color in zip(all_rewards.items(), COLORS):
    ax1.plot(rewards, color=color, alpha=0.15, linewidth=0.8)
    smoothed = np.convolve(rewards, np.ones(WINDOW) / WINDOW, mode="valid")
    ax1.plot(range(WINDOW - 1, EPISODES), smoothed, color=color, linewidth=2, label=name)

ax1.axhline(-13, color="white", linewidth=0.8, linestyle="--", alpha=0.3)
ax1.set_title("Q-Learning on CliffWalking-v1  ·  Three Epsilon Strategies",
              color="#E8EAF0", fontsize=13)
ax1.set_ylabel("Total Reward", color="#C9CDD8")
ax1.legend(labelcolor="#C9CDD8", framealpha=0.2, edgecolor="#2A2D3A")
ax1.grid(axis="y", color="#2A2D3A", linewidth=0.6)
ax1.tick_params(colors="#7A7F8E")

eps_schedules = [
    np.full(EPISODES, 0.5),
    np.full(EPISODES, 0.05),
    [max(0.01, 1.0 * (0.995 ** ep)) for ep in range(EPISODES)],
]
for sched, color in zip(eps_schedules, COLORS):
    ax2.plot(sched, color=color, linewidth=1.8)

ax2.set_ylabel("ε (epsilon)", color="#C9CDD8")
ax2.set_xlabel("Episode", color="#C9CDD8")
ax2.set_ylim(-0.05, 1.05)
ax2.grid(axis="y", color="#2A2D3A", linewidth=0.6)
ax2.tick_params(colors="#7A7F8E")

plt.tight_layout()
plt.savefig("cliffwalking_comparison.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.show()