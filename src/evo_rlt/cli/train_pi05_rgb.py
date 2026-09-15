"""Train PI05 with the OpenPI discrete_state_input=False behavior.

The switch is stored in the serialized RGB tokenizer step, not PI05Config.
All remaining CLI arguments are standard lerobot_train arguments.
"""


def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--discrete-state-input", choices=("false",), default="false")
    _, remaining = parser.parse_known_args()
    sys.argv[1:] = remaining
    from evo_rlt.adapters.lerobot.policies.processor_pi05_rgb import enable_rgb_training

    enable_rgb_training()
    from lerobot.scripts.lerobot_train import main as train

    train()


if __name__ == "__main__":
    main()
