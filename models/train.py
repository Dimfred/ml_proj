from bdqn import BDQNetwork, BDQConfig, ActivationFunction, Aggregator


def main():
    config = BDQConfig(
        n_outputs=1,
        shared_dim=[512, 256],
        actions_dim=[128],
        state_dim=[128],
        activation=ActivationFunction.from_str("ReLU"),
        n_action_branches=1,
        n_actions=33,
        aggregator=Aggregator.from_str("reduce_local_mean"),
    )
    model = BDQNetwork(config)

    model()


if __name__ == "__main__":
    main()
