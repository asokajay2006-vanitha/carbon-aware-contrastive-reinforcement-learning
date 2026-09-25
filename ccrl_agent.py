"""
Carbon-Aware Contrastive Reinforcement Learning (CCRL) Agent.

Architecture:

Raw Environment State
        ↓
Contrastive Encoder
        ↓
Learned Representation
        ↓
PPO Actor-Critic
        ↓
Resource Allocation Action

The CCRL agent combines:
    1. Contrastive representation learning
    2. PPO-based reinforcement learning
"""

import numpy as np
import torch

from contrastive import ContrastiveLearner
from ppo_agent import PPOAgent


class CCRLAgent:
    """
    Carbon-Aware Contrastive Reinforcement Learning agent.

    The raw environment state is first transformed into a
    lower-dimensional representation using contrastive learning.
    The learned representation is then given to the PPO agent.
    """

    def __init__(
        self,
        state_size,
        action_size,
        representation_size=64,
        hidden_size=128,
        learning_rate=3e-4,
        contrastive_learning_rate=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coefficient=0.01,
        value_coefficient=0.5,
        update_epochs=10,
        contrastive_temperature=0.07,
    ):

        self.state_size = state_size
        self.action_size = action_size
        self.representation_size = representation_size

        # ---------------------------------------------------------
        # Contrastive representation learning
        # ---------------------------------------------------------

        self.contrastive_learner = ContrastiveLearner(
            input_size=state_size,
            representation_size=representation_size,
            hidden_size=hidden_size,
            learning_rate=contrastive_learning_rate,
            temperature=contrastive_temperature,
        )

        # ---------------------------------------------------------
        # PPO reinforcement learning
        # ---------------------------------------------------------

        # IMPORTANT:
        # PPOAgent uses state_dim, action_dim and hidden_dim.
        self.ppo_agent = PPOAgent(
            state_dim=representation_size,
            action_dim=action_size,
            hidden_dim=hidden_size,
            learning_rate=learning_rate,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_epsilon=clip_epsilon,
            entropy_coefficient=entropy_coefficient,
            value_coefficient=value_coefficient,
            update_epochs=update_epochs,
        )

    # =============================================================
    # State Encoding
    # =============================================================

    def encode_state(self, state):
        """
        Convert a raw environment state into a learned
        contrastive representation.
        """

        state = np.asarray(
            state,
            dtype=np.float32,
        )

        representation = self.contrastive_learner.encode(
            state
        )

        return np.asarray(
            representation,
            dtype=np.float32,
        )

    # =============================================================
    # Action Selection
    # =============================================================

    def select_action(self, state):
        """
        Select a resource allocation action.

        Workflow:

        Raw state
            ↓
        Contrastive encoder
            ↓
        64-dimensional representation
            ↓
        PPO
            ↓
        Resource action
        """

        representation = self.encode_state(state)

        (
            action,
            log_prob,
            value,
        ) = self.ppo_agent.select_action(
            representation
        )

        return (
            action,
            log_prob,
            value,
            representation,
        )

    # =============================================================
    # Store Transition
    # =============================================================

    def store_transition(
        self,
        representation,
        action,
        reward,
        done,
        log_prob,
        value,
    ):
        """
        Store a PPO transition.
        """

        self.ppo_agent.store_transition(
            state=representation,
            action=action,
            reward=reward,
            done=done,
            log_prob=log_prob,
            value=value,
        )

    # =============================================================
    # PPO Update
    # =============================================================

    def update_policy(self):
        return self.ppo_agent.update()

    # =============================================================
    # Contrastive Training
    # =============================================================

    def train_contrastive(
        self,
        states,
        noise_scale=0.01,
    ):
        """
        Train the contrastive representation learner.

        Two augmented views of the same states are created
        and used for contrastive learning.
        """

        states = np.asarray(
            states,
            dtype=np.float32,
        )

        states_tensor = torch.tensor(
            states,
            dtype=torch.float32,
        )

        noise_1 = (
            torch.randn_like(states_tensor)
            * noise_scale
        )

        noise_2 = (
            torch.randn_like(states_tensor)
            * noise_scale
        )

        view_1 = states_tensor + noise_1
        view_2 = states_tensor + noise_2

        loss = self.contrastive_learner.train_step(
            view_1,
            view_2,
        )

        return loss

    # =============================================================
    # Save Model
    # =============================================================

    def save(self, path):
        """
        Save the CCRL model components.
        """

        checkpoint = {
            "contrastive_encoder":
                self.contrastive_learner.encoder.state_dict(),

            "contrastive_projector":
                self.contrastive_learner.projection_head.state_dict(),

            "ppo_policy":
                self.ppo_agent.policy.state_dict(),

            "ppo_optimizer":
                self.ppo_agent.optimizer.state_dict(),
        }

        torch.save(
            checkpoint,
            path,
        )

    # =============================================================
    # Load Model
    # =============================================================

    def load(self, path):
        """
        Load a previously saved CCRL model.
        """

        checkpoint = torch.load(
            path,
            map_location=self.ppo_agent.device,
        )

        self.contrastive_learner.encoder.load_state_dict(
            checkpoint["contrastive_encoder"]
        )

        self.contrastive_learner.projection_head.load_state_dict(
            checkpoint["contrastive_projector"]
        )

        self.ppo_agent.policy.load_state_dict(
            checkpoint["ppo_policy"]
        )

        self.ppo_agent.optimizer.load_state_dict(
            checkpoint["ppo_optimizer"]
        )


# =================================================================
# TEST
# =================================================================

if __name__ == "__main__":

    print("CCRL Agent Test")
    print("================")

    state_size = 153
    action_size = 25
    representation_size = 64

    print(
        f"Raw state size: {state_size}"
    )

    print(
        f"Action size: {action_size}"
    )

    print(
        f"Representation size: "
        f"{representation_size}"
    )

    # -------------------------------------------------------------
    # Create CCRL agent
    # -------------------------------------------------------------

    agent = CCRLAgent(
        state_size=state_size,
        action_size=action_size,
        representation_size=representation_size,
    )

    print()
    print("CCRL agent created successfully.")

    # -------------------------------------------------------------
    # Create sample state
    # -------------------------------------------------------------

    sample_state = np.random.rand(
        state_size
    ).astype(np.float32)

    print()
    print(
        f"Input state shape: "
        f"{sample_state.shape}"
    )

    # -------------------------------------------------------------
    # Select action
    # -------------------------------------------------------------

    (
        action,
        log_prob,
        value,
        representation,
    ) = agent.select_action(
        sample_state
    )

    print()
    print("Action selection")
    print("----------------")

    print(
        f"Selected action: {action}"
    )

    print(
        f"Log probability: "
        f"{log_prob:.6f}"
    )

    print(
        f"State value: "
        f"{value:.6f}"
    )

    print(
        f"Representation shape: "
        f"{representation.shape}"
    )

    # -------------------------------------------------------------
    # Test contrastive training
    # -------------------------------------------------------------

    sample_states = np.random.rand(
        16,
        state_size,
    ).astype(np.float32)

    contrastive_loss = (
        agent.train_contrastive(
            sample_states
        )
    )

    print()
    print("Contrastive learning")
    print("--------------------")

    if hasattr(
        contrastive_loss,
        "item",
    ):
        print(
            f"Contrastive loss: "
            f"{contrastive_loss.item():.6f}"
        )
    else:
        print(
            f"Contrastive loss: "
            f"{contrastive_loss:.6f}"
        )

    print()
    print(
        "CCRL agent test completed "
        "successfully."
    )