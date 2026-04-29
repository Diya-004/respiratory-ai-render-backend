from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Model, layers

from resp_ai.config import TrainConfig


def build_model(architecture: str, input_shape: tuple[int, int, int], num_classes: int, config: TrainConfig) -> Model:
    if architecture == "baseline_cnn":
        return build_baseline_cnn(input_shape, num_classes, config)
    if architecture == "custom_cnn":
        return build_custom_cnn(input_shape, num_classes, config)
    if architecture == "strong_cnn":
        return build_strong_cnn(input_shape, num_classes, config)
    if architecture == "efficientnetb0":
        return build_efficientnet(input_shape, num_classes, config)
    if architecture == "cnn_lstm":
        return build_cnn_lstm(input_shape, num_classes, config)
    raise ValueError(f"Unsupported architecture: {architecture}")


def build_baseline_cnn(input_shape: tuple[int, int, int], num_classes: int, config: TrainConfig) -> Model:
    inputs = layers.Input(shape=input_shape)
    x = inputs

    for filters in [32, 64, 128, 256]:
        x = layers.Conv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(256, 3, padding="same", activation="relu", name="gradcam_conv")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(config.dropout)(x)
    x = layers.Dense(config.dense_units, activation="relu")(x)
    x = layers.Dropout(config.dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return Model(inputs, outputs, name="baseline_cnn")


def squeeze_excite_block(x: tf.Tensor, filters: int) -> tf.Tensor:
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Dense(max(filters // 8, 8), activation="swish")(se)
    se = layers.Dense(filters, activation="sigmoid")(se)
    se = layers.Reshape((1, 1, filters))(se)
    return layers.Multiply()([x, se])


def residual_spectrogram_block(x: tf.Tensor, filters: int, dropout: float, downsample: bool = True) -> tf.Tensor:
    stride = 2 if downsample else 1
    shortcut = x

    x = layers.SeparableConv2D(filters, 3, strides=stride, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)

    # Asymmetric kernels help separate temporal and frequency structure in spectrograms.
    x = layers.Conv2D(filters, (5, 1), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)
    x = layers.Conv2D(filters, (1, 5), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    x = squeeze_excite_block(x, filters)

    if downsample or int(shortcut.shape[-1]) != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding="same", use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.Activation("swish")(x)
    if dropout > 0:
        x = layers.SpatialDropout2D(dropout)(x)
    return x


def build_custom_cnn(input_shape: tuple[int, int, int], num_classes: int, config: TrainConfig) -> Model:
    inputs = layers.Input(shape=input_shape)
    x = layers.BatchNormalization()(inputs)

    x = layers.Conv2D(32, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)

    x = residual_spectrogram_block(x, 32, dropout=config.dropout * 0.2, downsample=False)
    x = residual_spectrogram_block(x, 64, dropout=config.dropout * 0.25, downsample=True)
    x = residual_spectrogram_block(x, 96, dropout=config.dropout * 0.3, downsample=True)
    x = residual_spectrogram_block(x, 128, dropout=config.dropout * 0.35, downsample=True)

    x = layers.SeparableConv2D(192, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish", name="gradcam_conv")(x)

    avg_pool = layers.GlobalAveragePooling2D()(x)
    max_pool = layers.GlobalMaxPooling2D()(x)
    x = layers.Concatenate()([avg_pool, max_pool])
    x = layers.Dropout(config.dropout)(x)
    x = layers.Dense(config.dense_units, activation="swish")(x)
    x = layers.Dropout(config.dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return Model(inputs, outputs, name="custom_cnn")


def multiscale_residual_block(x: tf.Tensor, filters: int, dropout: float, stride: int = 1) -> tf.Tensor:
    shortcut = x

    branch_3x3 = layers.SeparableConv2D(filters, 3, strides=stride, padding="same", use_bias=False)(x)
    branch_3x3 = layers.BatchNormalization()(branch_3x3)
    branch_3x3 = layers.Activation("swish")(branch_3x3)

    branch_5x5 = layers.SeparableConv2D(filters, 5, strides=stride, padding="same", use_bias=False)(x)
    branch_5x5 = layers.BatchNormalization()(branch_5x5)
    branch_5x5 = layers.Activation("swish")(branch_5x5)

    branch_tf = layers.Conv2D(filters, (7, 1), strides=(stride, 1), padding="same", use_bias=False)(x)
    branch_tf = layers.BatchNormalization()(branch_tf)
    branch_tf = layers.Activation("swish")(branch_tf)
    branch_tf = layers.Conv2D(filters, (1, 7), strides=(1, stride), padding="same", use_bias=False)(branch_tf)
    branch_tf = layers.BatchNormalization()(branch_tf)
    branch_tf = layers.Activation("swish")(branch_tf)

    x = layers.Concatenate()([branch_3x3, branch_5x5, branch_tf])
    x = layers.Conv2D(filters, 1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = squeeze_excite_block(x, filters)

    if stride != 1 or int(shortcut.shape[-1]) != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding="same", use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.Activation("swish")(x)
    if dropout > 0:
        x = layers.SpatialDropout2D(dropout)(x)
    return x


def build_strong_cnn(input_shape: tuple[int, int, int], num_classes: int, config: TrainConfig) -> Model:
    inputs = layers.Input(shape=input_shape)
    x = layers.BatchNormalization()(inputs)

    x = layers.Conv2D(32, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)
    x = layers.Conv2D(32, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)

    x = multiscale_residual_block(x, 48, dropout=config.dropout * 0.15, stride=1)
    x = multiscale_residual_block(x, 96, dropout=config.dropout * 0.20, stride=2)
    x = multiscale_residual_block(x, 160, dropout=config.dropout * 0.25, stride=2)
    x = multiscale_residual_block(x, 224, dropout=config.dropout * 0.30, stride=2)
    x = multiscale_residual_block(x, 256, dropout=config.dropout * 0.35, stride=2)

    x = layers.SeparableConv2D(320, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish", name="gradcam_conv")(x)

    avg_pool = layers.GlobalAveragePooling2D()(x)
    max_pool = layers.GlobalMaxPooling2D()(x)
    x = layers.Concatenate()([avg_pool, max_pool])
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(config.dropout)(x)
    x = layers.Dense(config.dense_units, activation="swish")(x)
    x = layers.Dropout(config.dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return Model(inputs, outputs, name="strong_cnn")


def build_efficientnet(input_shape: tuple[int, int, int], num_classes: int, config: TrainConfig) -> Model:
    inputs = layers.Input(shape=input_shape)
    try:
        base = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_tensor=inputs,
            input_shape=input_shape,
        )
    except Exception:
        base = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights=None,
            input_tensor=inputs,
            input_shape=input_shape,
        )
    base.trainable = not config.freeze_backbone

    x = base.output
    x = layers.Conv2D(320, 1, activation="relu", name="gradcam_conv")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(config.dropout)(x)
    x = layers.Dense(config.dense_units, activation="relu")(x)
    x = layers.Dropout(config.dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return Model(inputs, outputs, name="efficientnetb0")


def build_cnn_lstm(input_shape: tuple[int, int, int], num_classes: int, config: TrainConfig) -> Model:
    inputs = layers.Input(shape=input_shape)
    x = inputs

    for filters in [32, 64, 128]:
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu", name="gradcam_conv")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Permute((2, 1, 3))(x)
    x = layers.Reshape((x.shape[1], x.shape[2] * x.shape[3]))(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=False))(x)
    x = layers.Dropout(config.dropout)(x)
    x = layers.Dense(config.dense_units, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return Model(inputs, outputs, name="cnn_lstm")
