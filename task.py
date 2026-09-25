from dataclasses import dataclass


@dataclass
class Task:
    """
    Represents a computational task in the edge-cloud system.
    """

    task_id: int
    input_size: float
    workload: float
    deadline: float
    arrival_time: float


def create_task(
    task_id: int,
    input_size: float,
    workload: float,
    deadline: float,
    arrival_time: float
) -> Task:
    """
    Create and return a Task object.
    """

    return Task(
        task_id=task_id,
        input_size=input_size,
        workload=workload,
        deadline=deadline,
        arrival_time=arrival_time
    )


if __name__ == "__main__":

    task = create_task(
        task_id=1,
        input_size=10.0,
        workload=500.0,
        deadline=100.0,
        arrival_time=0.0
    )

    print("Task created successfully")
    print("--------------------------------")
    print(f"Task ID       : {task.task_id}")
    print(f"Input size    : {task.input_size}")
    print(f"Workload      : {task.workload}")
    print(f"Deadline      : {task.deadline}")
    print(f"Arrival time  : {task.arrival_time}")