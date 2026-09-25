class RewardCalculator:
    def __init__(
        self,
        latency_weight=1.0,
        carbon_weight=1.0,
        qos_weight=1.0,
        carbon_awareness_weight=1.0,
    ):
        self.latency_weight = latency_weight
        self.carbon_weight = carbon_weight
        self.qos_weight = qos_weight
        self.carbon_awareness_weight = carbon_awareness_weight

    def calculate_latency_reward(self, latency):
        """Lower latency produces a better reward."""
        latency = max(0.0, float(latency))
        return -self.latency_weight * latency

    def calculate_carbon_reward(self, carbon_emissions):
        """Lower carbon emissions produce a better reward."""
        carbon_emissions = max(0.0, float(carbon_emissions))
        return -self.carbon_weight * carbon_emissions

    def calculate_qos_reward(self, qos_violation):
        """
        QoS violation is treated as a penalty.

        0 = No violation
        1 = Violation
        """
        qos_violation = max(0.0, float(qos_violation))
        return -self.qos_weight * qos_violation

    def calculate_carbon_awareness_reward(self, carbon_intensity):
        """Lower carbon intensity produces a better reward."""
        carbon_intensity = max(0.0, float(carbon_intensity))
        return -self.carbon_awareness_weight * carbon_intensity

    def calculate_reward(
        self,
        latency,
        carbon_emissions,
        qos_violation,
        carbon_intensity,
    ):
        """Calculate the total reward."""

        latency_reward = self.calculate_latency_reward(latency)

        carbon_reward = self.calculate_carbon_reward(
            carbon_emissions
        )

        qos_reward = self.calculate_qos_reward(
            qos_violation
        )

        carbon_awareness_reward = (
            self.calculate_carbon_awareness_reward(
                carbon_intensity
            )
        )

        total_reward = (
            latency_reward
            + carbon_reward
            + qos_reward
            + carbon_awareness_reward
        )

        reward_components = {
            "latency": latency_reward,
            "carbon": carbon_reward,
            "qos_violation": qos_reward,
            "carbon_awareness": carbon_awareness_reward,
        }

        return total_reward, reward_components


if __name__ == "__main__":

    print("Reward Calculator Test")
    print("======================")

    reward_calculator = RewardCalculator(
        latency_weight=1.0,
        carbon_weight=1.0,
        qos_weight=1.0,
        carbon_awareness_weight=1.0,
    )

    latency = 50.0
    carbon_emissions = 20.0
    qos_violation = 0.0
    carbon_intensity = 30.0

    total_reward, components = reward_calculator.calculate_reward(
        latency=latency,
        carbon_emissions=carbon_emissions,
        qos_violation=qos_violation,
        carbon_intensity=carbon_intensity,
    )

    print(f"Latency: {latency}")
    print(f"Carbon emissions: {carbon_emissions}")
    print(f"QoS violation: {qos_violation}")
    print(f"Carbon intensity: {carbon_intensity}")

    print()
    print("Reward components:")

    for name, value in components.items():
        print(f"  {name}: {value:.4f}")

    print()
    print(f"Total reward: {total_reward:.4f}")

