import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    """
    PPO Actor-Critic network.

    Input:
        64-dimensional state representation
        produced by the contrastive encoder.

    Output:
        Actor  -> probability distribution over 25 resources
        Critic -> scalar state-value estimate
    """

    def __init__(
        self,
        state_dim=64,
        action_dim=25,
        hidden_dim=128
    ):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor: selects a resource
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        # Critic: estimates the value of the state
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        features = self.shared(state)

        logits = self.actor(features)
        value = self.critic(features)

        return logits, value

    def get_action_distribution(self, state):
        logits, value = self.forward(state)

        distribution = Categorical(
            logits=logits
        )

        return distribution, value


class RolloutBuffer:
    """
    Stores transitions collected during PPO interaction.
    """

    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def add(
        self,
        state,
        action,
        log_prob,
        reward,
        done,
        value
    ):
        self.states.append(
            np.asarray(state, dtype=np.float32)
        )

        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()

    def __len__(self):
        return len(self.states)


class PPOAgent:
    """
    Proximal Policy Optimisation (PPO) agent.

    The agent:
        1. Receives a state representation.
        2. Selects one of 25 resource actions.
        3. Stores the transition.
        4. Calculates advantages using GAE.
        5. Updates the policy using the PPO clipped objective.
    """

    def __init__(
        self,
        state_dim=64,
        action_dim=25,
        hidden_dim=128,
        learning_rate=0.0003,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coefficient=0.5,
        entropy_coefficient=0.01,
        update_epochs=4,
        batch_size=64
    ):

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon

        self.value_coefficient = value_coefficient
        self.entropy_coefficient = entropy_coefficient

        self.update_epochs = update_epochs
        self.batch_size = batch_size

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.policy = ActorCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=learning_rate
        )

        self.buffer = RolloutBuffer()

    # --------------------------------------------------
    # Select action
    # --------------------------------------------------

    def select_action(self, state):
        """
        Select an action using the current policy.

        Returns:
            action
            log probability
            value estimate
        """

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        with torch.no_grad():

            distribution, value = (
                self.policy.get_action_distribution(
                    state_tensor
                )
            )

            action = distribution.sample()

            log_prob = distribution.log_prob(
                action
            )

        return (
            action.item(),
            log_prob.item(),
            value.item()
        )

    # --------------------------------------------------
    # Store transition
    # --------------------------------------------------

    def store_transition(
        self,
        state,
        action,
        log_prob,
        reward,
        done,
        value
    ):

        self.buffer.add(
            state=state,
            action=action,
            log_prob=log_prob,
            reward=reward,
            done=done,
            value=value
        )

    # --------------------------------------------------
    # Generalised Advantage Estimation
    # --------------------------------------------------

    def compute_gae(self):

        rewards = np.asarray(
            self.buffer.rewards,
            dtype=np.float32
        )

        dones = np.asarray(
            self.buffer.dones,
            dtype=np.float32
        )

        values = np.asarray(
            self.buffer.values,
            dtype=np.float32
        )

        advantages = np.zeros_like(
            rewards,
            dtype=np.float32
        )

        gae = 0.0

        for step in reversed(
            range(len(rewards))
        ):

            if step == len(rewards) - 1:
                next_value = 0.0
            else:
                next_value = values[step + 1]

            not_done = 1.0 - dones[step]

            delta = (
                rewards[step]
                + self.gamma
                * next_value
                * not_done
                - values[step]
            )

            gae = (
                delta
                + self.gamma
                * self.gae_lambda
                * not_done
                * gae
            )

            advantages[step] = gae

        returns = advantages + values

        return advantages, returns

    # --------------------------------------------------
    # PPO update
    # --------------------------------------------------

    def update(self):

        if len(self.buffer) == 0:
            raise ValueError(
                "Cannot update PPO with an empty buffer."
            )

        advantages, returns = (
            self.compute_gae()
        )

        states = torch.tensor(
            np.asarray(self.buffer.states),
            dtype=torch.float32,
            device=self.device
        )

        actions = torch.tensor(
            self.buffer.actions,
            dtype=torch.long,
            device=self.device
        )

        old_log_probs = torch.tensor(
            self.buffer.log_probs,
            dtype=torch.float32,
            device=self.device
        )

        advantages = torch.tensor(
            advantages,
            dtype=torch.float32,
            device=self.device
        )

        returns = torch.tensor(
            returns,
            dtype=torch.float32,
            device=self.device
        )

        # Normalise advantages
        advantages = (
            advantages
            - advantages.mean()
        ) / (
            advantages.std() + 1e-8
        )

        dataset_size = states.size(0)

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        update_count = 0

        for _ in range(self.update_epochs):

            indices = torch.randperm(
                dataset_size,
                device=self.device
            )

            for start in range(
                0,
                dataset_size,
                self.batch_size
            ):

                batch_indices = indices[
                    start:start + self.batch_size
                ]

                batch_states = states[
                    batch_indices
                ]

                batch_actions = actions[
                    batch_indices
                ]

                batch_old_log_probs = (
                    old_log_probs[
                        batch_indices
                    ]
                )

                batch_advantages = (
                    advantages[
                        batch_indices
                    ]
                )

                batch_returns = (
                    returns[
                        batch_indices
                    ]
                )

                distribution, values = (
                    self.policy
                    .get_action_distribution(
                        batch_states
                    )
                )

                new_log_probs = (
                    distribution.log_prob(
                        batch_actions
                    )
                )

                entropy = (
                    distribution.entropy()
                )

                # Convert critic output from
                # [batch_size, 1] to [batch_size]
                values = values.squeeze(-1)

                # PPO probability ratio
                ratios = torch.exp(
                    new_log_probs
                    - batch_old_log_probs
                )

                # Clipped objective
                surrogate_1 = (
                    ratios
                    * batch_advantages
                )

                surrogate_2 = (
                    torch.clamp(
                        ratios,
                        1.0 - self.clip_epsilon,
                        1.0 + self.clip_epsilon
                    )
                    * batch_advantages
                )

                actor_loss = -torch.min(
                    surrogate_1,
                    surrogate_2
                ).mean()

                # Critic loss
                # Both values and batch_returns now have
                # the same shape: [batch_size]
                values = values.squeeze(-1)
                critic_loss = nn.functional.mse_loss(
                    values,
                    batch_returns
                )

                entropy_bonus = entropy.mean()

                loss = (
                    actor_loss
                    + self.value_coefficient
                    * critic_loss
                    - self.entropy_coefficient
                    * entropy_bonus
                )

                self.optimizer.zero_grad()

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    max_norm=0.5
                )

                self.optimizer.step()

                total_actor_loss += (
                    actor_loss.item()
                )

                total_critic_loss += (
                    critic_loss.item()
                )

                total_entropy += (
                    entropy_bonus.item()
                )

                update_count += 1

        self.buffer.clear()

        return {
            "actor_loss":
                total_actor_loss / update_count,

            "critic_loss":
                total_critic_loss / update_count,

            "entropy":
                total_entropy / update_count
        }


# ======================================================
# TEST
# ======================================================

if __name__ == "__main__":

    print("PPO Test")
    print("================================")

    # --------------------------------------------------
    # PPO configuration
    # --------------------------------------------------

    state_dim = 64
    action_dim = 25

    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=128,
        learning_rate=0.0003,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        update_epochs=4,
        batch_size=16
    )

    print(
        f"State dimension : {state_dim}"
    )

    print(
        f"Action dimension: {action_dim}"
    )

    print(
        f"Device          : {agent.device}"
    )

    # --------------------------------------------------
    # Generate test transitions
    # --------------------------------------------------

    number_of_steps = 64

    for step in range(
        number_of_steps
    ):

        state = np.random.randn(
            state_dim
        ).astype(np.float32)

        action, log_prob, value = (
            agent.select_action(state)
        )

        reward = np.random.uniform(
            -10.0,
            1.0
        )

        done = (
            step == number_of_steps - 1
        )

        agent.store_transition(
            state=state,
            action=action,
            log_prob=log_prob,
            reward=reward,
            done=done,
            value=value
        )

    print(
        f"\nTransitions stored: "
        f"{len(agent.buffer)}"
    )

    # --------------------------------------------------
    # Test one action distribution
    # --------------------------------------------------

    test_state = np.random.randn(
        state_dim
    ).astype(np.float32)

    action, log_prob, value = (
        agent.select_action(test_state)
    )

    print("\nAction selection test")
    print("--------------------------------")

    print(
        f"Selected action       : {action}"
    )

    print(
        f"Log probability       : "
        f"{log_prob:.6f}"
    )

    print(
        f"Value estimate        : "
        f"{value:.6f}"
    )

    # --------------------------------------------------
    # PPO update
    # --------------------------------------------------

    update_result = agent.update()

    print("\nPPO update completed")
    print("--------------------------------")

    print(
        f"Actor loss            : "
        f"{update_result['actor_loss']:.6f}"
    )

    print(
        f"Critic loss           : "
        f"{update_result['critic_loss']:.6f}"
    )

    print(
        f"Entropy               : "
        f"{update_result['entropy']:.6f}"
    )

    print("\nPPO test completed successfully.")