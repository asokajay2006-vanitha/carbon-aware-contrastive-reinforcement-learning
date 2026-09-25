import random
import numpy as np
from reward import RewardCalculator
from task import create_task
from resource import create_all_resources
from carbon_model import calculate_resource_carbon


class EdgeCloudEnvironment:
    """
    Simulation environment for distributed edge-cloud
    resource scheduling.
    """

    def __init__(
        self,
        number_of_edges=20,
        number_of_cloud=5,
        task_rate=50,
        max_steps=100
    ):
        """
        Initialise the edge-cloud environment.

        Parameters
        ----------
        number_of_edges : int
            Number of edge nodes.

        number_of_cloud : int
            Number of cloud servers.

        task_rate : int
            Approximate task arrival rate.

        max_steps : int
            Maximum number of scheduling decisions
            in one episode.
        """

        self.number_of_edges = number_of_edges
        self.number_of_cloud = number_of_cloud
        self.task_rate = task_rate
        self.max_steps = max_steps

        self.resources = create_all_resources()

        self.number_of_resources = len(self.resources)

        self.current_step = 0
        self.current_task = None

        self.total_latency = 0.0
        self.total_energy = 0.0
        self.total_carbon = 0.0
        self.qos_violations = 0

    def generate_task(self):
        """
        Generate a new computational task.
        """

        task_id = self.current_step

        input_size = random.uniform(1.0, 20.0)

        workload = random.uniform(
            100.0,
            1000.0
        )

        deadline = random.uniform(
            50.0,
            200.0
        )

        arrival_time = float(self.current_step)

        return create_task(
            task_id=task_id,
            input_size=input_size,
            workload=workload,
            deadline=deadline,
            arrival_time=arrival_time
        )

    def calculate_network_latency(self, resource):
        """
        Estimate network latency based on resource type
        and bandwidth.
        """

        if resource.resource_type == "edge":
            base_latency = 5.0
        else:
            base_latency = 20.0

        bandwidth_factor = (
            100.0 / resource.bandwidth
        )

        latency = (
            base_latency
            + 5.0 * bandwidth_factor
        )

        return latency

    def build_state(self):
        """
        Construct the environment state.

        The state contains:
        - task workload
        - task input size
        - task deadline
        - resource CPU utilisation
        - available CPU
        - bandwidth
        - CPU frequency
        - power
        - carbon intensity
        """

        state = []

        if self.current_task is None:
            return np.zeros(
                3 + (self.number_of_resources * 6),
                dtype=np.float32
            )

        state.extend([
            self.current_task.input_size,
            self.current_task.workload,
            self.current_task.deadline
        ])

        for resource in self.resources:

            state.extend([
                resource.utilisation,
                resource.available_cpu(),
                resource.bandwidth,
                resource.cpu_frequency,
                resource.power,
                resource.carbon_intensity
            ])

        return np.array(
            state,
            dtype=np.float32
        )

    def reset(self):
        """
        Start a new episode.
        """

        self.current_step = 0

        self.total_latency = 0.0
        self.total_energy = 0.0
        self.total_carbon = 0.0
        self.qos_violations = 0

        self.resources = create_all_resources()

        self.current_task = self.generate_task()

        state = self.build_state()

        return state

    def step(self, action):
        """
        Execute one scheduling action.

        Parameters
        ----------
        action : int
            Resource selected by the agent.

        Returns
        -------
        next_state : numpy.ndarray
        reward : float
        done : bool
        info : dict
        """

        if not isinstance(action, (int, np.integer)):
            raise TypeError(
                "Action must be an integer."
            )

        if action < 0 or action >= self.number_of_resources:
            raise ValueError(
                f"Action must be between 0 and "
                f"{self.number_no_resources - 1}."
            )

        resource = self.resources[action]

        task = self.current_task

        # Calculate available CPU
        available_cpu = resource.available_cpu()

        # Avoid zero or negative computational capacity
        effective_cpu = max(
            available_cpu,
            0.1
        )

        # Calculate task execution information
        result = calculate_resource_carbon(
            workload=task.workload,
            cpu_capacity=effective_cpu,
            power_watts=resource.power,
            carbon_intensity=resource.carbon_intensity
        )

        execution_time = result["execution_time"]

        energy = result["energy_wh"]

        carbon = result["carbon_grams"]

        # Calculate network latency
        network_latency = self.calculate_network_latency(
            resource
        )

        # Total latency
        total_latency = (
            execution_time
            + network_latency
        )

        # QoS check
        qos_violation = (
            total_latency > task.deadline
        )

        if qos_violation:
            self.qos_violations += 1

        # Update resource utilisation
        utilisation_increase = min(
            task.workload / (
                resource.cpu_capacity * 1000.0
            ),
            0.20
        )

        resource.utilisation = min(
            resource.utilisation
            + utilisation_increase,
            1.0
        )

        # Update cumulative metrics
        self.total_latency += total_latency
        self.total_energy += energy
        self.total_carbon += carbon

        # Basic reward
        reward = -(
            total_latency
            + carbon
        )

        # Additional QoS penalty
        if qos_violation:
            reward -= 100.0

        # Move to next step
        self.current_step += 1

        done = (
            self.current_step >= self.max_steps
        )

        # Generate next task
        if not done:
            self.current_task = self.generate_task()

        next_state = self.build_state()

        info = {
            "task_id": task.task_id,
            "resource_id": resource.resource_id,
            "resource_type": resource.resource_type,
            "latency": total_latency,
            "energy_wh": energy,
            "carbon_grams": carbon,
            "qos_violation": qos_violation,
            "total_latency": self.total_latency,
            "total_energy": self.total_energy,
            "total_carbon": self.total_carbon,
            "qos_violations": self.qos_violations
        }

        return (
            next_state,
            reward,
            done,
            info
        )


if __name__ == "__main__":

    print("Edge-Cloud Environment Test")
    print("================================")

    env = EdgeCloudEnvironment(
        number_of_edges=20,
        number_of_cloud=5,
        task_rate=50,
        max_steps=5
    )

    print(
        f"Number of resources: "
        f"{env.number_of_resources}"
    )

    state = env.reset()

    print(
        f"Initial state size: "
        f"{len(state)}"
    )

    print(
        f"Initial task ID: "
        f"{env.current_task.task_id}"
    )

    print(
        f"Initial workload: "
        f"{env.current_task.workload:.2f}"
    )

    print("\nRunning environment steps...")
    print("--------------------------------")

    for step in range(5):

        # Test action: select resource 0
        action = 0

        next_state, reward, done, info = env.step(
            action
        )

        print(
            f"Step {step + 1}: "
            f"Resource={info['resource_id']}, "
            f"Type={info['resource_type']}, "
            f"Latency={info['latency']:.2f}, "
            f"Energy={info['energy_wh']:.4f} Wh, "
            f"Carbon={info['carbon_grams']:.4f} gCO2, "
            f"QoS violation={info['qos_violation']}, "
            f"Reward={reward:.2f}"
        )

        if done:
            break

    print("\nEpisode summary")
    print("--------------------------------")

    print(
        f"Total latency : "
        f"{env.total_latency:.2f}"
    )

    print(
        f"Total energy  : "
        f"{env.total_energy:.4f} Wh"
    )

    print(
        f"Total carbon  : "
        f"{env.total_carbon:.4f} gCO2"
    )

    print(
        f"QoS violations: "
        f"{env.qos_violations}"
    )
