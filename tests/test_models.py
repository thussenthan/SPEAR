from spear.config import TrainingConfig
from spear.models import TorchModelBundle, build_model


def _peak_distance_training_config() -> TrainingConfig:
    return TrainingConfig(
        per_gene_feature_basis="peak",
        per_gene_peak_distance_encoding="rbf",
        per_gene_peak_distance_rbf_bases=16,
    )


def test_flat_cnn_and_resnet_do_not_pick_up_peak_distance_channels() -> None:
    config = _peak_distance_training_config()

    cnn_bundle = build_model("cnn", 84, config)
    assert isinstance(cnn_bundle, TorchModelBundle)
    assert cnn_bundle.reshape == "flat"
    assert cnn_bundle.model.backbone[0].in_channels == 1

    resnet_bundle = build_model("resnet", 84, config)
    assert isinstance(resnet_bundle, TorchModelBundle)
    assert resnet_bundle.reshape == "flat"
    assert resnet_bundle.model.stem[0].in_channels == 1


def test_sequence_models_pick_up_peak_distance_channels() -> None:
    config = _peak_distance_training_config()

    rnn_bundle = build_model("rnn", 84, config)
    assert isinstance(rnn_bundle, TorchModelBundle)
    assert rnn_bundle.reshape == "sequence"
    assert rnn_bundle.model.project[0].in_channels == 18

    lstm_bundle = build_model("lstm", 84, config)
    assert isinstance(lstm_bundle, TorchModelBundle)
    assert lstm_bundle.reshape == "sequence"
    assert lstm_bundle.model.project[0].in_channels == 18

    transformer_bundle = build_model("transformer", 84, config)
    assert isinstance(transformer_bundle, TorchModelBundle)
    assert transformer_bundle.reshape == "sequence"
    assert transformer_bundle.model.project[0].in_channels == 18
