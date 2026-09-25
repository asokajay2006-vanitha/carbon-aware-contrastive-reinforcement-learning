"""
Results logger for the Carbon-Aware Contrastive
Reinforcement Learning project.

This module saves training results to CSV files.
"""

import csv
import os


class ResultsLogger:
    """
    Saves episode-level training results.
    """

    def __init__(
        self,
        results_directory="results",
    ):
        self.results_directory = (
            results_directory
        )

        os.makedirs(
            self.results_directory,
            exist_ok=True,
        )

        self.results_file = os.path.join(
            self.results_directory,
            "training_results.csv",
        )

        self.results = []

    def log_episode(
        self,
        episode,
        steps,
        reward,
        latency,
        carbon,
        qos_violation,
    ):
        """
        Store results for one training episode.
        """

        result = {
            "episode": episode,
            "steps": steps,
            "reward": reward,
            "latency": latency,
            "carbon": carbon,
            "qos_violation": qos_violation,
        }

        self.results.append(result)

    def save(self):
        """
        Save all recorded results to CSV.
        """

        if not self.results:
            print(
                "No results available to save."
            )
            return

        fieldnames = [
            "episode",
            "steps",
            "reward",
            "latency",
            "carbon",
            "qos_violation",
        ]

        with open(
            self.results_file,
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            writer.writerows(
                self.results
            )

        print(
            f"Training results saved to: "
            f"{self.results_file}"
        )

    def clear(self):
        """
        Clear results from memory.
        """

        self.results.clear()


if __name__ == "__main__":

    print("Results Logger Test")
    print("===================")

    logger = ResultsLogger(
        "results"
    )

    logger.log_episode(
        episode=1,
        steps=10,
        reward=25.50,
        latency=45.20,
        carbon=18.30,
        qos_violation=0.0,
    )

    logger.log_episode(
        episode=2,
        steps=10,
        reward=28.40,
        latency=42.10,
        carbon=16.70,
        qos_violation=0.0,
    )

    logger.save()

    print(
        "Results logger test completed."
    )