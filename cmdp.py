class CMDPController:
    """
    Constrained Markov Decision Process (CMDP) controller.

    Constraints:
        1. Latency
        2. Carbon emissions
        3. QoS violation
        4. Resource utilisation
    """

    def __init__(
        self,
        latency_limit=100.0,
        carbon_limit=100.0,
        qos_limit=0.0,
        resource_utilisation_limit=1.0,
        learning_rate=0.01
    ):

        if latency_limit <= 0:
            raise ValueError(
                "Latency limit must be greater than zero."
            )

        if carbon_limit <= 0:
            raise ValueError(
                "Carbon limit must be greater than zero."
            )

        if qos_limit < 0:
            raise ValueError(
                "QoS limit cannot be negative."
            )

        if not 0.0 < resource_utilisation_limit <= 1.0:
            raise ValueError(
                "Resource utilisation limit must be between 0 and 1."
            )

        if learning_rate <= 0:
            raise ValueError(
                "Learning rate must be greater than zero."
            )

        self.latency_limit = latency_limit
        self.carbon_limit = carbon_limit
        self.qos_limit = qos_limit
        self.resource_utilisation_limit = (
            resource_utilisation_limit
        )
        self.learning_rate = learning_rate

        # Lagrange multipliers
        self.lambda_latency = 1.0
        self.lambda_carbon = 1.0
        self.lambda_qos = 1.0
        self.lambda_resource = 1.0

    # --------------------------------------------------
    # Constraint violation functions
    # --------------------------------------------------

    def latency_violation(self, latency):
        return max(
            0.0,
            latency - self.latency_limit
        )

    def carbon_violation(self, carbon):
        return max(
            0.0,
            carbon - self.carbon_limit
        )

    def qos_violation(self, qos):
        return max(
            0.0,
            qos - self.qos_limit
        )

    def resource_violation(self, utilisation):
        return max(
            0.0,
            utilisation - self.resource_utilisation_limit
        )

    # --------------------------------------------------
    # Evaluate constraints
    # --------------------------------------------------

    def evaluate_constraints(
        self,
        latency,
        carbon,
        qos,
        resource_utilisation
    ):

        violations = {
            "latency": self.latency_violation(latency),
            "carbon": self.carbon_violation(carbon),
            "qos": self.qos_violation(qos),
            "resource": self.resource_violation(
                resource_utilisation
            )
        }

        status = {
            key: (
                "Satisfied"
                if value == 0.0
                else "Violated"
            )
            for key, value in violations.items()
        }

        feasible = all(
            value == 0.0
            for value in violations.values()
        )

        return {
            "violations": violations,
            "status": status,
            "feasible": feasible,
            "overall_feasible": feasible
        }

    # --------------------------------------------------
    # Lagrangian penalty
    # --------------------------------------------------

    def calculate_lagrangian_penalty(
        self,
        violations
    ):

        penalty = (
            self.lambda_latency
            * violations["latency"]
            +
            self.lambda_carbon
            * violations["carbon"]
            +
            self.lambda_qos
            * violations["qos"]
            +
            self.lambda_resource
            * violations["resource"]
        )

        return penalty

    # --------------------------------------------------
    # Constrained reward
    # --------------------------------------------------

    def calculate_constrained_reward(
        self,
        reward,
        violations
    ):

        penalty = self.calculate_lagrangian_penalty(
            violations
        )

        return reward - penalty

    # --------------------------------------------------
    # Update Lagrange multipliers
    # --------------------------------------------------

    def update_multipliers(
        self,
        violations
    ):

        self.lambda_latency += (
            self.learning_rate
            * violations["latency"]
        )

        self.lambda_carbon += (
            self.learning_rate
            * violations["carbon"]
        )

        self.lambda_qos += (
            self.learning_rate
            * violations["qos"]
        )

        self.lambda_resource += (
            self.learning_rate
            * violations["resource"]
        )

        # Keep multipliers within a stable range.
        self.lambda_latency = min(
            10.0,
            max(
                0.0,
                self.lambda_latency
            )
        )

        self.lambda_carbon = min(
            10.0,
            max(
                0.0,
                self.lambda_carbon
            )
        )

        self.lambda_qos = min(
            10.0,
            max(
                0.0,
                self.lambda_qos
            )
        )

        self.lambda_resource = min(
            10.0,
            max(
                0.0,
                self.lambda_resource
            )
        )

    # --------------------------------------------------
    # Complete CMDP processing
    # --------------------------------------------------

    def process(
        self,
        reward,
        latency,
        carbon,
        qos,
        resource_utilisation
    ):

        result = self.evaluate_constraints(
            latency=latency,
            carbon=carbon,
            qos=qos,
            resource_utilisation=resource_utilisation
        )

        violations = result["violations"]

        constrained_reward = (
            self.calculate_constrained_reward(
                reward=reward,
                violations=violations
            )
        )

        self.update_multipliers(
            violations
        )

        return {
            "violations": violations,
            "status": result["status"],
            "feasible": result["feasible"],
            "constrained_reward": constrained_reward,
            "lambda_latency": self.lambda_latency,
            "lambda_carbon": self.lambda_carbon,
            "lambda_qos": self.lambda_qos,
            "lambda_resource": self.lambda_resource
        }


# ======================================================
# TEST
# ======================================================

if __name__ == "__main__":

    print("CMDP Constraint Test")
    print("====================")

    cmdp = CMDPController(
        latency_limit=100.0,
        carbon_limit=100.0,
        qos_limit=0.0,
        resource_utilisation_limit=1.0,
        learning_rate=0.01
    )

    latency = 75.0
    carbon = 60.0
    qos = 0.0
    resource_utilisation = 0.8

    base_reward = -135.0

    result = cmdp.process(
        reward=base_reward,
        latency=latency,
        carbon=carbon,
        qos=qos,
        resource_utilisation=resource_utilisation
    )

    print(f"Latency: {latency}")
    print(f"Carbon emissions: {carbon}")
    print(f"QoS violation: {qos}")
    print(
        f"Resource utilization: "
        f"{resource_utilisation}"
    )

    print("\nConstraint status:")

    for name, status in result["status"].items():
        print(
            f"  {name}: {status}"
        )

    print(
        f"\nOverall feasible: "
        f"{result['feasible']}"
    )

    print("\nConstraint violations:")

    for name, violation in result["violations"].items():
        print(
            f"  {name}: "
            f"{violation:.4f}"
        )

    print("\nReward information:")

    print(
        f"  Base reward: "
        f"{base_reward:.4f}"
    )

    print(
        f"  Constrained reward: "
        f"{result['constrained_reward']:.4f}"
    )

    print("\nLagrange multipliers:")

    print(
        f"  Lambda latency: "
        f"{result['lambda_latency']:.4f}"
    )

    print(
        f"  Lambda carbon: "
        f"{result['lambda_carbon']:.4f}"
    )

    print(
        f"  Lambda QoS: "
        f"{result['lambda_qos']:.4f}"
    )

    print(
        f"  Lambda resource: "
        f"{result['lambda_resource']:.4f}"
    )