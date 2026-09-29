"""LeRobot PI05 training with valid-action-only flow-matching loss."""


def main():
    from evo_rlt.adapters.lerobot.policies.pi05_masked_loss import enable_masked_pi05_training

    enable_masked_pi05_training()
    from lerobot.scripts.lerobot_train import main as train

    train()


if __name__ == "__main__":
    main()
