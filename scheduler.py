"""
Carbon-Aware Contrastive Reinforcement Learning (CCRL) Scheduler.

Workflow:

Environment
    ↓
Raw State
    ↓
Contrastive Encoder
    ↓
Learned Representation
    ↓
PPO Actor-Critic
    ↓
Resource Allocation Action
    ↓
Environment
    ↓
Latency / Carbon / Energy / QoS
    ↓
CMDP Constraint Evaluation
    ↓
Constraint Violations
    ↓
Lagrangian Penalty
    ↓
Constrained Reward
    ↓
PPO Update
"""

from environment import EdgeCloudEnvironment
from ccrl_agent import CCRLAgent
from cmdp import CMDPController


class CCRLScheduler:
    """
    Main scheduler for the Carbon-Aware Contrastive
    Reinforcement Learning framework.
    """

    def __init__(
        self,
        state_size=153,
        action_size=25,
        max_steps=100,
    ):

        self.state_size = state_size
        self.action_size = action_size
        self.max_steps = max_steps

        # ---------------------------------------------------------
        # Environment
        # ---------------------------------------------------------

        self.environment = EdgeCloudEnvironment(
            number_of_edges=20,
            number_of_cloud=5,
            task_rate=50,
            max_steps=max_steps,
        )

        # ---------------------------------------------------------
        # CCRL Agent
        # ---------------------------------------------------------

        self.agent = CCRLAgent(
            state_size=state_size,
            action_size=action_size,
            representation_size=64,
        )

        # ---------------------------------------------------------
        # CMDP Controller
        # ---------------------------------------------------------

        self.cmdp = CMDPController(
            latency_limit=100.0,
            carbon_limit=100.0,
            qos_limit=0.0,
            learning_rate=0.01,
        )

    # =============================================================
    # Run One Episode
    # =============================================================

    def run_episode(self, training=True):
        """
        Run one complete CCRL scheduling episode.
        """

        state = self.environment.reset()

        total_base_reward = 0.0
        total_constrained_reward = 0.0

        total_latency = 0.0
        total_carbon = 0.0
        total_energy = 0.0

        qos_violations = 0
        constraint_violations = 0

        step_count = 0
        contrastive_states = []
        ppo_update = {
            "actor_loss": 0.0,
            "critic_loss": 0.0,
            "entropy": 0.0,
         }
        while True:

            # -----------------------------------------------------
            # 1. Select action using CCRL
            # -----------------------------------------------------

            (
                action,
                log_prob,
                value,
                representation,
            ) = self.agent.select_action(state)

            # -----------------------------------------------------
            # 2. Execute action in environment
            # -----------------------------------------------------

            (
                next_state,
                base_reward,
                done,
                info,
            ) = self.environment.step(action)

            # -----------------------------------------------------
            # 3. Read environment metrics
            # -----------------------------------------------------

            latency = float(
                info.get(
                    "latency",
                    0.0,
                )
            )

            carbon = float(
                info.get("carbon_grams", 0.0))

            energy = float(
                info.get("energy_wh", 0.0))

            qos_violation = bool(
                info.get(
                    "qos_violation",
                    False,
                )
            )

            utilisation = float(
                info.get(
                    "utilisation",
                    0.0,
                )
            )

            # -----------------------------------------------------
            # 4. Evaluate CMDP constraints
            # -----------------------------------------------------

            constraint_result = (
                self.cmdp.evaluate_constraints(
                    latency,
                    carbon,
                    float(qos_violation),
                    utilisation,
                )
            )

            # -----------------------------------------------------
            # 5. Extract constraint violations
            #
            # CMDP returns the violations inside:
            #
            # constraint_result["violations"]
            # -----------------------------------------------------

            violations = (
                constraint_result["violations"]
            )

            # -----------------------------------------------------
            # 6. Calculate Lagrangian penalty
            # -----------------------------------------------------

            lagrangian_penalty = (
                self.cmdp.calculate_lagrangian_penalty(
                    violations
                )
            )

            # -----------------------------------------------------
            # 7. Calculate constrained reward
            # -----------------------------------------------------

            constrained_reward = (
                base_reward
                - lagrangian_penalty
            )

            # -----------------------------------------------------
            # 8. Update Lagrange multipliers
            # -----------------------------------------------------

            self.cmdp.update_multipliers(
                violations
            )

            # -----------------------------------------------------
            # 9. Store transition for PPO
            # -----------------------------------------------------

            if training:

                self.agent.store_transition(
                    representation=representation,
                    action=action,
                    reward=constrained_reward,
                    done=done,
                    log_prob=log_prob,
                    value=value,
                )
                contrastive_states.append(state)

            # -----------------------------------------------------
            # 10. Accumulate metrics
            # -----------------------------------------------------

            total_base_reward += base_reward

            total_constrained_reward += (
                constrained_reward
            )

            total_latency += latency
            total_carbon += carbon
            total_energy += energy

            if qos_violation:
                qos_violations += 1

            if not constraint_result.get(
                "overall_feasible",
                True,
            ):
                constraint_violations += 1

            step_count += 1

            state = next_state

            # -----------------------------------------------------
            # 11. End episode
            # -----------------------------------------------------

            if done:
                break

        # ---------------------------------------------------------
        # 12. Update PPO
        # ---------------------------------------------------------

        update_result = None

        if training and len(contrastive_states) >= 2:
            contrastive_loss = self.agent.train_contrastive(
                contrastive_states
            )
        else:
            contrastive_loss = None

        if training:
            ppo_update = self.agent.update_policy()           
        # ---------------------------------------------------------
        # 13. Return episode results
        # ---------------------------------------------------------
        return {
            "steps":
                step_count,

            "base_reward":
                total_base_reward,

            "constrained_reward":
                total_constrained_reward,

            "latency":
                total_latency,

            "carbon":
                total_carbon,

            "energy":
                total_energy,

            "qos_violations":
                qos_violations,

            "constraint_violations":
                constraint_violations,

            "contrastive_loss":
                contrastive_loss,

            "update_result":
                update_result,
        }

    # =============================================================
    # Training
    # =============================================================

    def train(
        self,
        number_of_episodes=10,
    ):
        """
        Train the CCRL scheduler.
        """

        history = []

        print("CCRL Training")
        print("================================")
        print(
            f"Episodes      : "
            f"{number_of_episodes}"
        )
        print(
            f"Maximum steps : "
            f"{self.max_steps}"
        )

        for episode in range(
            1,
            number_of_episodes + 1,
        ):

            results = self.run_episode(
                training=True
            )

            history.append(results)

            print()
            print(
                f"Episode {episode}"
            )

            print(
                f"  Steps              : "
                f"{results['steps']}"
            )
            print(
               f"Contrastive loss   : "
               f"{results['contrastive_loss']}"
            )

            print(
                f"  Base reward        : "
                f"{results['base_reward']:.4f}"
            )

            print(
                f"  Constrained reward : "
                f"{results['constrained_reward']:.4f}"
            )

            print(
                f"  Latency            : "
                f"{results['latency']:.4f}"
            )

            print(
                f"  Carbon             : "
                f"{results['carbon']:.4f}"
            )

            print(
                f"  Energy             : "
                f"{results['energy']:.4f}"
            )

            print(
                f"  QoS violations     : "
                f"{results['qos_violations']}"
            )

            print(
                f"  Constraint errors  : "
                f"{results['constraint_violations']}"
            )

        return history


# =================================================================
# TEST
# =================================================================

if __name__ == "__main__":

    print("CCRL Scheduler Test")
    print("==============================")

    # -------------------------------------------------------------
    # Create scheduler
    # -------------------------------------------------------------

    scheduler = CCRLScheduler(
        state_size=153,
        action_size=25,
        max_steps=10,
    )

    print(
        f"State size       : "
        f"{scheduler.state_size}"
    )

    print(
        f"Action size      : "
        f"{scheduler.action_size}"
    )

    print(
        f"Maximum steps    : "
        f"{scheduler.max_steps}"
    )

    print()
    print(
        "CMDP controller created."
    )

    # -------------------------------------------------------------
    # Run one test episode
    # -------------------------------------------------------------

    print()
    print(
        "Running one test episode..."
    )

    results = scheduler.run_episode(
        training=True
    )

    # -------------------------------------------------------------
    # Display results
    # -------------------------------------------------------------

    print()
    print("Episode results")
    print("------------------------------")

    print(
        f"Steps              : "
        f"{results['steps']}"
    )
    print(
        f"Contrastive loss   : "
        f"{results['contrastive_loss']}"
    )
    print(
        f"Base reward        : "
        f"{results['base_reward']:.4f}"
    )

    print(
        f"Constrained reward : "
        f"{results['constrained_reward']:.4f}"
    )

    print(
        f"Total latency      : "
        f"{results['latency']:.4f}"
    )

    print(
        f"Total carbon       : "
        f"{results['carbon']:.4f}"
    )

    print(
        f"Total energy       : "
        f"{results['energy']:.4f}"
    )

    print(
        f"QoS violations     : "
        f"{results['qos_violations']}"
    )

    print(
        f"Constraint errors  : "
        f"{results['constraint_violations']}"
    )

    print()
    print(
        "CCRL scheduler test completed "
        "successfully."
    )