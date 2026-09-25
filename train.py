"""
Configuration-based training pipeline for
Carbon-Aware Contrastive Reinforcement Learning.

Architecture:

config.yaml
    ↓
ConfigLoader
    ↓
Environment
    ↓
CCRL Agent
    ↓
Reward
    ↓
CMDP Constraints
    ↓
PPO Update
"""

import os
import random
import numpy as np

from config_loader import ConfigLoader
from environment import EdgeCloudEnvironment
from reward import RewardCalculator
from cmdp import CMDPConstraints
from ccrl_agent import CCRLAgent
from results_logger import ResultsLogger

class CCRLTrainer:
    """
    Main training manager for the CCRL framework.
    """

    def __init__(self):

        # -------------------------------------------------
        # Load configuration
        # -------------------------------------------------

        self.config = ConfigLoader()

        # -------------------------------------------------
        # Environment configuration
        # -------------------------------------------------

        environment_config = (
            self.config.get_section(
                "environment"
            )
        )

        self.state_size = int(
            environment_config.get(
                "state_size",
                153,
            )
        )

        self.action_size = int(
            environment_config.get(
                "action_size",
                25,
            )
        )

        # -------------------------------------------------
        # Training configuration
        # -------------------------------------------------

        training_config = (
            self.config.get_section(
                "training"
            )
        )

        self.episodes = int(
            training_config.get(
                "episodes",
                10,
            )
        )

        self.seed = int(
            training_config.get(
                "seed",
                42,
            )
        )

        # -------------------------------------------------
        # Contrastive configuration
        # -------------------------------------------------

        contrastive_config = (
            self.config.get_section(
                "contrastive"
            )
        )

        self.representation_size = int(
            contrastive_config.get(
                "representation_size",
                64,
            )
        )

        self.contrastive_hidden_size = int(
            contrastive_config.get(
                "hidden_size",
                128,
            )
        )

        self.contrastive_learning_rate = float(
            contrastive_config.get(
                "learning_rate",
                0.001,
            )
        )

        self.contrastive_temperature = float(
            contrastive_config.get(
                "temperature",
                0.07,
            )
        )

        # -------------------------------------------------
        # PPO configuration
        # -------------------------------------------------

        ppo_config = (
            self.config.get_section(
                "ppo"
            )
        )

        self.ppo_hidden_size = int(
            ppo_config.get(
                "hidden_size",
                128,
            )
        )

        self.ppo_learning_rate = float(
            ppo_config.get(
                "learning_rate",
                0.0003,
            )
        )

        self.gamma = float(
            ppo_config.get(
                "gamma",
                0.99,
            )
        )

        self.gae_lambda = float(
            ppo_config.get(
                "gae_lambda",
                0.95,
            )
        )

        self.clip_epsilon = float(
            ppo_config.get(
                "clip_epsilon",
                0.2,
            )
        )

        self.entropy_coefficient = float(
            ppo_config.get(
                "entropy_coefficient",
                0.01,
            )
        )

        self.value_coefficient = float(
            ppo_config.get(
                "value_coefficient",
                0.5,
            )
        )

        self.update_epochs = int(
            ppo_config.get(
                "update_epochs",
                10,
            )
        )

        # -------------------------------------------------
        # Reward configuration
        # -------------------------------------------------

        reward_config = (
            self.config.get_section(
                "reward"
            )
        )

        self.latency_weight = float(
            reward_config.get(
                "latency_weight",
                1.0,
            )
        )

        self.carbon_weight = float(
            reward_config.get(
                "carbon_weight",
                1.0,
            )
        )

        self.qos_weight = float(
            reward_config.get(
                "qos_weight",
                1.0,
            )
        )

        self.carbon_awareness_weight = float(
            reward_config.get(
                "carbon_awareness_weight",
                1.0,
            )
        )

        # -------------------------------------------------
        # CMDP configuration
        # -------------------------------------------------

        cmdp_config = (
            self.config.get_section(
                "cmdp"
            )
        )

        self.latency_limit = float(
            cmdp_config.get(
                "latency_limit",
                100.0,
            )
        )

        self.carbon_limit = float(
            cmdp_config.get(
                "carbon_limit",
                100.0,
            )
        )

        self.qos_limit = float(
            cmdp_config.get(
                "qos_limit",
                0.0,
            )
        )

        self.resource_utilization_limit = float(
            cmdp_config.get(
                "resource_utilization_limit",
                1.0,
            )
        )

        # -------------------------------------------------
        # Set random seed
        # -------------------------------------------------

        self.set_seed(
            self.seed
        )

        # -------------------------------------------------
        # Create environment
        # -------------------------------------------------

        self.environment = (
            EdgeCloudEnvironment()
        )

        # -------------------------------------------------
        # Create reward calculator
        # -------------------------------------------------

        self.reward_calculator = (
            RewardCalculator(
                latency_weight=self.latency_weight,
                carbon_weight=self.carbon_weight,
                qos_weight=self.qos_weight,
                carbon_awareness_weight=(
                    self.carbon_awareness_weight
                ),
            )
        )

        # -------------------------------------------------
        # Create CMDP
        # -------------------------------------------------

        self.cmdp = CMDPConstraints(
            latency_limit=self.latency_limit,
            carbon_limit=self.carbon_limit,
            qos_limit=self.qos_limit,
            resource_utilization_limit=(
                self.resource_utilization_limit
            ),
        )

        # -------------------------------------------------
        # Create CCRL agent
        # -------------------------------------------------

        self.agent = CCRLAgent(
            state_size=self.state_size,
            action_size=self.action_size,
            representation_size=(
                self.representation_size
            ),
            hidden_size=(
                self.ppo_hidden_size
            ),
            learning_rate=(
                self.ppo_learning_rate
            ),
            contrastive_learning_rate=(
                self.contrastive_learning_rate
            ),
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_epsilon=self.clip_epsilon,
            entropy_coefficient=(
                self.entropy_coefficient
            ),
            value_coefficient=(
                self.value_coefficient
            ),
            update_epochs=self.update_epochs,
            contrastive_temperature=(
                self.contrastive_temperature
            ),
        )

        # -------------------------------------------------
        # Training history
        # -------------------------------------------------

        @staticmethod= []
        self.episode_latencies = []
        self.episode_carbon = []
        self.episode_qos_violations = []
# -------------------------------------------------
# Results logger
# -------------------------------------------------

results_config = self.config.get_section(
    "results"
)

results_directory = results_config.get(
    "directory",
    "results",
)

self.results_logger = ResultsLogger(
    results_directory=results_directory
)
    # =====================================================
    # Random seed
    # =====================================================
        # Random seed
    @staticmethod
    def set_seed(seed):
        """
        Set random seeds for reproducibility.
        """

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    # =====================================================
    # Environment metrics
    # =====================================================

    def get_environment_metrics(
        self,
        info,
    ):

        latency = float(
            info.get(
                "latency",
                0.0,
            )
        )

        carbon_emissions = float(
            info.get(
                "carbon_emissions",
                info.get(
                    "carbon",
                    0.0,
                ),
            )
        )

        qos_violation = float(
            info.get(
                "qos_violation",
                0.0,
            )
        )

        resource_utilization = float(
            info.get(
                "resource_utilization",
                0.0,
            )
        )

        carbon_intensity = float(
            info.get(
                "carbon_intensity",
                0.0,
            )
        )

        return (
            latency,
            carbon_emissions,
            qos_violation,
            resource_utilization,
            carbon_intensity,
        )

    # =====================================================
    # Training
    # =====================================================

    def train(self):

        print()
        print("==========================================")
        print(
            " Carbon-Aware Contrastive RL Training"
        )
        print("==========================================")
        print()

        print(
            f"Configuration: "
            f"{self.config.config_path}"
        )

        print(
            f"Number of episodes: "
            f"{self.episodes}"
        )

        print(
            f"State size: "
            f"{self.state_size}"
        )

        print(
            f"Action size: "
            f"{self.action_size}"
        )

        print(
            f"Representation size: "
            f"{self.representation_size}"
        )

        print(
            f"Random seed: "
            f"{self.seed}"
        )

        print()

        for episode in range(
            1,
            self.episodes + 1,
        ):

            reset_result = (
                self.environment.reset()
            )

            if isinstance(
                reset_result,
                tuple,
            ):

                state = reset_result[0]

            else:

                state = reset_result

            state = np.asarray(
                state,
                dtype=np.float32,
            )

            done = False

            episode_reward = 0.0
            episode_latency = 0.0
            episode_carbon = 0.0
            episode_qos = 0.0

            step_count = 0

            while not done:

                # -----------------------------------------
                # Select action
                # -----------------------------------------

                (
                    action,
                    log_prob,
                    value,
                    representation,
                ) = self.agent.select_action(
                    state
                )

                # -----------------------------------------
                # Environment step
                # -----------------------------------------

                step_result = (
                    self.environment.step(
                        action
                    )
                )

                if len(step_result) == 5:

                    (
                        next_state,
                        environment_reward,
                        terminated,
                        truncated,
                        info,
                    ) = step_result

                    done = (
                        terminated
                        or truncated
                    )

                elif len(step_result) == 4:

                    (
                        next_state,
                        environment_reward,
                        done,
                        info,
                    ) = step_result

                else:

                    raise ValueError(
                        "Unexpected number of "
                        "values returned by "
                        "environment.step()."
                    )

                next_state = np.asarray(
                    next_state,
                    dtype=np.float32,
                )

                # -----------------------------------------
                # Metrics
                # -----------------------------------------

                (
                    latency,
                    carbon_emissions,
                    qos_violation,
                    resource_utilization,
                    carbon_intensity,
                ) = self.get_environment_metrics(
                    info
                )

                # -----------------------------------------
                # Reward
                # -----------------------------------------

                reward, reward_components = (
                    self.reward_calculator.calculate_reward(
                        latency=latency,
                        carbon_emissions=carbon_emissions,
                        qos_violation=qos_violation,
                        carbon_intensity=carbon_intensity,
                    )
                )

                # -----------------------------------------
                # CMDP constraints
                # -----------------------------------------

                constraints, feasible = (
                    self.cmdp.check_all_constraints(
                        latency=latency,
                        carbon_emissions=(
                            carbon_emissions
                        ),
                        qos_violation=qos_violation,
                        resource_utilization=(
                            resource_utilization
                        ),
                    )
                )

                # -----------------------------------------
                # Store transition
                # -----------------------------------------

                self.agent.store_transition(
                    representation=representation,
                    action=action,
                    reward=reward,
                    done=done,
                    log_prob=log_prob,
                    value=value,
                )

                # -----------------------------------------
                # Statistics
                # -----------------------------------------

                episode_reward += reward
                episode_latency += latency
                episode_carbon += carbon_emissions
                episode_qos += qos_violation

                step_count += 1

                state = next_state

            # ---------------------------------------------
            # PPO update
            # ---------------------------------------------

            self.agent.update_policy()

            # ---------------------------------------------
            # Save statistics
            # ---------------------------------------------

            self.episode_rewards.append(
                episode_reward
            )

            self.episode_latencies.append(
                episode_latency
            )

            self.episode_carbon.append(
                episode_carbon
            )

            self.episode_qos_violations.append(
                episode_qos
            )

            print(
                f"Episode {episode:03d} | "
                f"Steps: {step_count:03d} | "
                f"Reward: {episode_reward:10.4f} | "
                f"Latency: {episode_latency:10.4f} | "
                f"Carbon: {episode_carbon:10.4f} | "
                f"QoS: {episode_qos:8.4f}"
            )

        print()
        print(
            "=========================================="
        )
        print(
            " Training completed successfully."
        )
        print(
            "=========================================="
        )

        return {
            "rewards": self.episode_rewards,
            "latencies": self.episode_latencies,
            "carbon": self.episode_carbon,
            "qos_violations": (
                self.episode_qos_violations
            ),
        }

    # =====================================================
    # Save model
    # =====================================================

    def save_model(self):

        training_config = (
            self.config.get_section(
                "training"
            )
        )

        save_model = training_config.get(
            "save_model",
            True,
        )

        model_path = training_config.get(
            "model_path",
            "models/ccrl_model.pth",
        )

        if not save_model:

            print(
                "Model saving is disabled "
                "in config.yaml."
            )

            return

        directory = os.path.dirname(
            model_path
        )

        if directory:

            os.makedirs(
                directory,
                exist_ok=True,
            )

        self.agent.save(
            model_path
        )

    # =====================================================
    # Training summary
    # =====================================================

    def print_summary(self):

        if not self.episode_rewards:

            print(
                "No training results available."
            )

            return

        print()
        print("Training Summary")
        print("================")

        print(
            f"Average reward: "
            f"{np.mean(self.episode_rewards):.4f}"
        )

        print(
            f"Average latency: "
            f"{np.mean(self.episode_latencies):.4f}"
        )

        print(
            f"Average carbon: "
            f"{np.mean(self.episode_carbon):.4f}"
        )

        print(
            f"Average QoS violation: "
            f"{np.mean(self.episode_qos_violations):.4f}"
        )


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    trainer = CCRLTrainer()

    trainer.train()

    trainer.print_summary()

    trainer.save_model()

    print()
    print(
        "CCRL training program completed."
    )