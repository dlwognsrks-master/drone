import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import config
import model_separated as model_lib


def main():
    art = config.ARTIFACTS_DIR

    X_past_train = np.load(f'{art}/X_past_train.npy')
    X_batt_state_train = np.load(f'{art}/X_batt_state_train.npy')
    y_rft_train = np.load(f'{art}/y_rft_train.npy')

    X_past_val = np.load(f'{art}/X_past_val.npy')
    X_batt_state_val = np.load(f'{art}/X_batt_state_val.npy')
    y_rft_val = np.load(f'{art}/y_rft_val.npy')

    model = model_lib.build_model(
        past_len=config.PAST_SEQ_LEN,
        past_features=X_past_train.shape[2],
        state_dim=X_batt_state_train.shape[1],
    )

    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(config.LEARNING_RATE),
        loss=keras.losses.Huber(),
    )

    cb = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=5),
        ModelCheckpoint(f'{art}/best_model.keras', save_best_only=True)
    ]

    train_x = [X_past_train, X_batt_state_train]
    val_x = [X_past_val, X_batt_state_val]

    model.fit(
        train_x,
        y_rft_train,
        validation_data=(val_x, y_rft_val),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=cb
    )

    print("Training Complete (RFT).")

    # 학습 곡선 시각화
    hist = model.history.history
    epochs = range(1, len(hist['loss']) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, hist['loss'], label='train')
    axes[0].plot(epochs, hist['val_loss'], label='val')
    axes[0].set_title('Total Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()

    axes[1].plot(epochs, hist['val_loss'], label='val')
    axes[1].set_title('Val Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f'{art}/learning_curve.png', dpi=150)
    plt.close()
    print(f"[INFO] 학습 곡선 저장: {art}/learning_curve.png")


if __name__ == "__main__":
    main()
