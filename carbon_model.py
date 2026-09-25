def calculate_execution_time(workload, cpu_capacity):
    """
    Calculate approximate execution time.

    Parameters
    ----------
    workload : float
        Computational workload.
    cpu_capacity : float
        Available CPU capacity.

    Returns
    -------
    float
        Execution time in seconds.
    """

    if workload < 0:
        raise ValueError("Workload cannot be negative.")

    if cpu_capacity <= 0:
        raise ValueError("CPU capacity must be greater than zero.")

    return workload / cpu_capacity


def calculate_energy(power_watts, execution_time_seconds):
    """
    Calculate energy consumption.

    Parameters
    ----------
    power_watts : float
        Power consumption in watts.
    execution_time_seconds : float
        Execution time in seconds.

    Returns
    -------
    float
        Energy consumption in Wh.
    """

    if power_watts < 0:
        raise ValueError("Power cannot be negative.")

    if execution_time_seconds < 0:
        raise ValueError(
            "Execution time cannot be negative."
        )

    execution_time_hours = execution_time_seconds / 3600.0

    energy_wh = power_watts * execution_time_hours

    return energy_wh


def calculate_carbon_emission(
    energy_wh,
    carbon_intensity
):
    """
    Calculate carbon emission.

    Parameters
    ----------
    energy_wh : float
        Energy consumption in Wh.
    carbon_intensity : float
        Carbon intensity in gCO2/kWh.

    Returns
    -------
    float
        Carbon emission in grams of CO2.
    """

    if energy_wh < 0:
        raise ValueError(
            "Energy consumption cannot be negative."
        )

    if carbon_intensity < 0:
        raise ValueError(
            "Carbon intensity cannot be negative."
        )

    energy_kwh = energy_wh / 1000.0

    carbon_grams = energy_kwh * carbon_intensity

    return carbon_grams


def calculate_resource_carbon(
    workload,
    cpu_capacity,
    power_watts,
    carbon_intensity
):
    """
    Calculate execution time, energy and carbon
    emission for a task-resource assignment.
    """

    execution_time = calculate_execution_time(
        workload,
        cpu_capacity
    )

    energy = calculate_energy(
        power_watts,
        execution_time
    )

    carbon = calculate_carbon_emission(
        energy,
        carbon_intensity
    )

    return {
        "execution_time": execution_time,
        "energy_wh": energy,
        "carbon_grams": carbon
    }


if __name__ == "__main__":

    print("Carbon model test")
    print("--------------------------------")

    workload = 500.0
    cpu_capacity = 5.0
    power_watts = 60.0
    carbon_intensity = 150.0

    result = calculate_resource_carbon(
        workload=workload,
        cpu_capacity=cpu_capacity,
        power_watts=power_watts,
        carbon_intensity=carbon_intensity
    )

    print(f"Workload          : {workload}")
    print(f"CPU capacity      : {cpu_capacity}")
    print(f"Power             : {power_watts} W")
    print(
        f"Carbon intensity  : "
        f"{carbon_intensity} gCO2/kWh"
    )

    print("\nCalculated values")
    print("--------------------------------")

    print(
        f"Execution time    : "
        f"{result['execution_time']:.2f} seconds"
    )

    print(
        f"Energy consumption: "
        f"{result['energy_wh']:.4f} Wh"
    )

    print(
        f"Carbon emission   : "
        f"{result['carbon_grams']:.4f} gCO2"
    )