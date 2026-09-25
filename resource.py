from dataclasses import dataclass


@dataclass
class Resource:
    """
    Represents a computing resource in the distributed
    edge-cloud system.
    """

    resource_id: int
    resource_type: str
    cpu_capacity: float
    bandwidth: float
    cpu_frequency: float
    power: float
    carbon_intensity: float
    utilisation: float = 0.0

    def available_cpu(self) -> float:
        """
        Calculate the currently available CPU capacity.
        """

        return self.cpu_capacity * (1.0 - self.utilisation)

    def update_utilisation(self, new_utilisation: float):
        """
        Update resource utilisation.

        The utilisation value must be between 0 and 1.
        """

        if not 0.0 <= new_utilisation <= 1.0:
            raise ValueError(
                "Utilisation must be between 0 and 1."
            )

        self.utilisation = new_utilisation


def create_edge_resources(number_of_edges: int = 20):
    """
    Create heterogeneous edge resources.
    """

    resources = []

    for i in range(number_of_edges):

        resource = Resource(
            resource_id=i,
            resource_type="edge",
            cpu_capacity=4.0 + (i % 4),
            bandwidth=100.0 + ((i % 5) * 50.0),
            cpu_frequency=2.0 + ((i % 3) * 0.5),
            power=50.0 + ((i % 4) * 10.0),
            carbon_intensity=100.0 + ((i % 6) * 50.0),
            utilisation=0.0
        )

        resources.append(resource)

    return resources


def create_cloud_resources(number_of_cloud: int = 5):
    """
    Create heterogeneous cloud resources.
    """

    resources = []

    for i in range(number_of_cloud):

        resource_id = 20 + i

        resource = Resource(
            resource_id=resource_id,
            resource_type="cloud",
            cpu_capacity=8.0 + (i * 2.0),
            bandwidth=500.0 + (i * 100.0),
            cpu_frequency=3.0 + (i * 0.25),
            power=120.0 + (i * 15.0),
            carbon_intensity=80.0 + (i * 40.0),
            utilisation=0.0
        )

        resources.append(resource)

    return resources


def create_all_resources():
    """
    Create the complete distributed edge-cloud system.
    """

    edge_resources = create_edge_resources(20)
    cloud_resources = create_cloud_resources(5)

    return edge_resources + cloud_resources


if __name__ == "__main__":

    resources = create_all_resources()

    print("Resources created successfully")
    print("--------------------------------")

    print(f"Total resources : {len(resources)}")

    edge_count = sum(
        1 for resource in resources
        if resource.resource_type == "edge"
    )

    cloud_count = sum(
        1 for resource in resources
        if resource.resource_type == "cloud"
    )

    print(f"Edge resources  : {edge_count}")
    print(f"Cloud resources : {cloud_count}")

    print("\nFirst five resources:")
    print("--------------------------------")

    for resource in resources[:5]:

        print(
            f"ID={resource.resource_id}, "
            f"Type={resource.resource_type}, "
            f"CPU={resource.cpu_capacity}, "
            f"Bandwidth={resource.bandwidth}, "
            f"Frequency={resource.cpu_frequency}, "
            f"Power={resource.power}, "
            f"Carbon={resource.carbon_intensity}"
        )

    print("\nTesting available CPU:")

    first_resource = resources[0]

    print(
        f"Resource {first_resource.resource_id} "
        f"available CPU = "
        f"{first_resource.available_cpu():.2f}"
    )

    first_resource.update_utilisation(0.50)

    print(
        f"Resource {first_resource.resource_id} "
        f"utilisation = "
        f"{first_resource.utilisation:.2f}"
    )

    print(
        f"Resource {first_resource.resource_id} "
        f"available CPU after utilisation update = "
        f"{first_resource.available_cpu():.2f}"
    )
    