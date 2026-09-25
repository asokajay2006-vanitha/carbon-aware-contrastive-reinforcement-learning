"""
Configuration loader for the CCRL project.

This module loads parameters from:

    config/config.yaml

and provides them to the training program.
"""

import os
import yaml


class ConfigLoader:
    """
    Loads and manages the CCRL configuration.
    """

    def __init__(self, config_path=None):

        if config_path is None:
            # Project root is one level above src/
            project_root = os.path.dirname(
                os.path.dirname(
                    os.path.abspath(__file__)
                )
            )

            config_path = os.path.join(
                project_root,
                "config",
                "config.yaml",
            )

        self.config_path = config_path

        self.config = self._load_config()

    def _load_config(self):
        """
        Read the YAML configuration file.
        """

        if not os.path.exists(
            self.config_path
        ):
            raise FileNotFoundError(
                f"Configuration file not found: "
                f"{self.config_path}"
            )

        with open(
            self.config_path,
            "r",
            encoding="utf-8",
        ) as file:

            config = yaml.safe_load(file)

        if config is None:
            raise ValueError(
                "The configuration file is empty."
            )

        return config

    def get(self, section, key, default=None):
        """
        Get a specific configuration value.

        Example:

            config.get(
                "environment",
                "state_size"
            )
        """

        section_data = self.config.get(
            section,
            {},
        )

        return section_data.get(
            key,
            default,
        )

    def get_section(self, section):
        """
        Return an entire configuration section.
        """

        return self.config.get(
            section,
            {},
        )

    def get_all(self):
        """
        Return the complete configuration.
        """

        return self.config


if __name__ == "__main__":

    print("Configuration Loader Test")
    print("=========================")

    config = ConfigLoader()

    print()
    print(
        f"Configuration file: "
        f"{config.config_path}"
    )

    print()
    print("Environment settings:")

    environment = config.get_section(
        "environment"
    )

    for key, value in environment.items():
        print(f"  {key}: {value}")

    print()
    print("Reward settings:")

    reward = config.get_section(
        "reward"
    )

    for key, value in reward.items():
        print(f"  {key}: {value}")

    print()
    print("PPO settings:")

    ppo = config.get_section(
        "ppo"
    )

    for key, value in ppo.items():
        print(f"  {key}: {value}")

    print()
    print(
        "Configuration loaded successfully."
    )