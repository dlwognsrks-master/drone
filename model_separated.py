import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class TemporalAttention(layers.Layer):
    """시계열 time step별 중요도 가중합 (weighted sum)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score_dense = layers.Dense(1)

    def call(self, x):
        score = self.score_dense(x)               # (batch, T, 1)
        weight = tf.nn.softmax(score, axis=1)     # (batch, T, 1)
        return tf.reduce_sum(x * weight, axis=1)  # (batch, C)

    def get_config(self):
        return super().get_config()


class MaskedGlobalAveragePooling1D(layers.Layer):
    """
    Mask-aware mean pooling.
    padding(0) 길이로 shortcut 학습되는 것을 막기 위해,
    유효 step만 평균에 반영한다.
    """

    def call(self, inputs, mask=None):
        x = inputs
        if mask is None:
            return tf.reduce_mean(x, axis=1)

        mask = tf.cast(mask, x.dtype)
        mask = tf.expand_dims(mask, axis=-1)
        x = x * mask
        denom = tf.reduce_sum(mask, axis=1) + tf.keras.backend.epsilon()
        return tf.reduce_sum(x, axis=1) / denom


def _build_encoder_scratch(past_len: int, past_features: int) -> keras.Model:
    """오토인코더 사전학습 없이 random init CNN 인코더 직접 생성"""
    inp = keras.Input(shape=(past_len, past_features), name="past_input_enc")
    x = layers.Conv1D(64, 5, padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Temporal Attention: 중요한 time step에 집중 (weighted sum) → (batch, 128)
    x = TemporalAttention(name='temporal_attn')(x)
    return keras.Model(inp, x, name="base_encoder")


def build_model(
    past_len: int,
    past_features: int,
    state_dim: int = 4,  # consumed_mah, V_ema, I_ema, sag
):
    """RFT (Remaining Flight Time) 예측 모델: CNN 인코더 + battery_state 멀티인풋"""

    input_past = keras.Input(shape=(past_len, past_features), name="past_input")
    input_state = keras.Input(shape=(state_dim,), name="state_input")

    base_encoder = _build_encoder_scratch(past_len, past_features)
    z_full = base_encoder(input_past)

    rft_x = layers.Concatenate(name="rft_concat")([z_full, input_state])

    rft_x = layers.Dense(128, name="rft_dense1")(rft_x)
    rft_x = layers.BatchNormalization(name="rft_bn1")(rft_x)
    rft_x = layers.Activation("relu", name="rft_relu1")(rft_x)
    rft_x = layers.Dropout(0.3, name="rft_dropout1")(rft_x)
    rft_x = layers.Dense(64, name="rft_dense2")(rft_x)
    rft_x = layers.BatchNormalization(name="rft_bn2")(rft_x)
    rft_x = layers.Activation("relu", name="rft_relu2")(rft_x)
    rft_x = layers.Dropout(0.2, name="rft_dropout2")(rft_x)
    rft_output = layers.Dense(1, name="rft_output")(rft_x)

    model = keras.Model(inputs=[input_past, input_state], outputs=rft_output, name="rft_model")
    return model
