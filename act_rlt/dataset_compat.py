"""Process-local compatibility for single-channel LeRobot robot states."""


def install_scalar_state_dataset_support():
    """Read a one-channel robot state as [1], while timestamps remain scalar.

    Affected LeRobot versions map every shape=(1,) feature to a scalar HF
    Value. Robot state is a vector even with one channel. Override only the
    reader's state schema, without changing installed package files or data.
    """
    import datasets
    from lerobot.datasets import dataset_reader

    original = dataset_reader.get_hf_features_from_features
    if getattr(original, '_scalar_state_vector', False):
        return

    def features_with_state_vector(features):
        result = original(features)
        state = features.get('observation.state')
        if state is not None and tuple(state['shape']) == (1,):
            result['observation.state'] = datasets.Sequence(
                feature=datasets.Value(dtype=state['dtype']), length=1)
        return result

    features_with_state_vector._scalar_state_vector = True
    dataset_reader.get_hf_features_from_features = features_with_state_vector
