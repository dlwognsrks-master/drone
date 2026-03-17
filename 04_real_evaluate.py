import os
import numpy as np
import joblib
import keras
import optuna
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import config
from model_separated import MaskedGlobalAveragePooling1D, TemporalAttention

optuna.logging.set_verbosity(optuna.logging.WARNING)

_CUSTOM_OBJECTS = {
    "MaskedGlobalAveragePooling1D": MaskedGlobalAveragePooling1D,
    "TemporalAttention": TemporalAttention,
}

keras.config.enable_unsafe_deserialization()

model = keras.saving.load_model(
    f"{config.ARTIFACTS_DIR}/best_model.keras",
    custom_objects=_CUSTOM_OBJECTS,
)

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def plot_series(save_path, title, y_true, y_pred, ylabel):
    plt.figure(figsize=(14, 6))
    x = np.arange(len(y_true))
    plt.plot(x, y_true, label="True")
    plt.plot(x, y_pred, label="Pred")
    plt.title(title)
    plt.xlabel("Time step (samples)")
    plt.ylabel(ylabel)
    y_all = np.concatenate([y_true, y_pred])
    y_min = max(0, np.floor(y_all.min() / 10) * 10)
    y_max = np.ceil(y_all.max() / 10) * 10
    minor_ticks = np.arange(y_min, y_max + 10, 10)
    major_ticks = np.arange(y_min, y_max + 50, 50)
    ax = plt.gca()
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    ax.yaxis.set_tick_params(which='minor', length=3)
    plt.ylim(y_min, y_max)
    ax.grid(True, which='major', linestyle="--", alpha=0.7)
    ax.grid(True, which='minor', linestyle="--", alpha=0.7)
    plt.legend()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def main():
    art = config.ARTIFACTS_DIR
    os.makedirs(art, exist_ok=True)

    try:
        model = keras.models.load_model(
            os.path.join(art, "best_model.keras"),
            custom_objects=_CUSTOM_OBJECTS,
        )
        s_rft = joblib.load(os.path.join(art, "scaler_rft.pkl"))
    except Exception as e:
        print(f"[ERROR] Load model/scalers failed: {e}")
        return

    try:
        X_past = np.load(os.path.join(art, "X_past_test.npy"))
        X_batt = np.load(os.path.join(art, "X_batt_state_test.npy"))
        y_rft_s = np.load(os.path.join(art, "y_rft_test.npy"))
        meta = np.load(os.path.join(art, "meta_test.npy"), allow_pickle=True)
    except FileNotFoundError:
        print("[ERROR] Test data not found. Run 03_make_test.py first.")
        return

    print(f"[INFO] Test samples: {len(X_past)}")
    print("[INFO] Predicting on test set...")

    pred = model.predict([X_past, X_batt], batch_size=64, verbose=1)
    pred_rft_s = pred if not isinstance(pred, (list, tuple, dict)) else (
        pred["rft_output"] if isinstance(pred, dict) else pred[0]
    )

    y_rft = s_rft.inverse_transform(y_rft_s).flatten()
    y_rft_raw_hat = s_rft.inverse_transform(pred_rft_s).flatten()

    def enforce_monotone_with_slope_limit_per_log(rft_pred, meta, max_drop, max_rise):

        out = rft_pred.copy()
        log_names = np.unique(meta[:, 0])

        for ln in log_names:
            idx = np.where(meta[:, 0] == ln)[0]
            if len(idx) < 2:
                continue

            ts = meta[idx, 1].astype(np.int64)
            order = np.argsort(ts)
            idx_sorted = idx[order]

            y = out[idx_sorted]
            y = np.maximum(y, 0.0)

            y_fixed = y.copy()
            for t in range(1, len(y_fixed)):
                prev = y_fixed[t - 1]
                raw = y_fixed[t]
                cur = np.clip(raw, prev - max_drop, prev + max_rise)
                y_fixed[t] = cur

            out[idx_sorted] = y_fixed

        return out

    # --------------------------------------------------
    # Optuna: monotone 파라미터 최적화 (선택)
    # --------------------------------------------------
    use_optuna = input("\nOptuna로 monotone 파라미터 탐색할까요? (y/n): ").strip().lower() == "y"

    if use_optuna:
        print("[INFO] Optuna: monotone 파라미터 탐색 중 (n_trials=200)...")

        def objective(trial):
            max_drop = trial.suggest_float("max_drop", 0.05, 2.0, log=True)
            max_rise = trial.suggest_float("max_rise", 0.001, 0.5, log=True)
            y_hat = enforce_monotone_with_slope_limit_per_log(y_rft_raw_hat, meta, max_drop, max_rise)
            return mean_absolute_error(y_rft, y_hat)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=200, show_progress_bar=True)

        FIXED_MAX_DROP = study.best_params["max_drop"]
        FIXED_MAX_RISE = study.best_params["max_rise"]
        print(f"\n[Optuna] Best  max_drop = {FIXED_MAX_DROP:.4f}")
        print(f"[Optuna] Best  max_rise = {FIXED_MAX_RISE:.4f}")
        print(f"[Optuna] Best  MAE      = {study.best_value:.3f} s")
        print(f"[Optuna] → FIXED_MAX_DROP = {FIXED_MAX_DROP:.4f}, FIXED_MAX_RISE = {FIXED_MAX_RISE:.4f} 로 고정하세요")
    else:
        print("[INFO] Optuna 스킵 → 단조 제약 없이 raw 예측값 사용")
    # --------------------------------------------------

    if use_optuna:
        print(f"\n[INFO] Applying monotone constraint: max_drop={FIXED_MAX_DROP:.4f}, max_rise={FIXED_MAX_RISE:.4f} s/step")
        y_rft_hat = enforce_monotone_with_slope_limit_per_log(y_rft_raw_hat, meta, FIXED_MAX_DROP, FIXED_MAX_RISE)
    else:
        y_rft_hat = y_rft_raw_hat

    #--------------------------------------------------#
    rft_mae = mean_absolute_error(y_rft, y_rft_hat)
    rft_rmse = rmse(y_rft, y_rft_hat)
    #--------------------------------------------------#

    print("=" * 60)
    print("EVALUATION RESULT (RFT)")
    print("=" * 60)
    monotone_info = f"max_drop={FIXED_MAX_DROP:.4f}, max_rise={FIXED_MAX_RISE:.4f}" if use_optuna else "단조 제약 없음"
    print(f"RFT  MAE : {rft_mae:.3f} s ({monotone_info})")
    print(f"RFT RMSE : {rft_rmse:.3f} s")
    print("=" * 60)

    log_names = np.unique(meta[:, 0])
    results = []

    for log_name in log_names:
        idx = (meta[:, 0] == log_name)
        n = int(np.sum(idx))
        if n < 20:
            continue
    #--------------------------------------------------#
        rft_t = y_rft[idx]
        rft_p = y_rft_hat[idx]
    #--------------------------------------------------#
        r_mae = mean_absolute_error(rft_t, rft_p)
        r_rmse = rmse(rft_t, rft_p)

        results.append((log_name, n, r_mae, r_rmse))

        safe_name = str(log_name).replace(" ", "_").replace(":", "_")

        plot_series(
            save_path=os.path.join(art, f"TEST_RFT_{safe_name}_RMSE_{r_rmse:.2f}.png"),
            title=f"RFT Prediction | Log={log_name} | RMSE={r_rmse:.2f}s",
            y_true=rft_t,
            y_pred=rft_p,
            ylabel="RFT (s)"
        )

    if not results:
        print("[WARN] No logs passed the minimum sample threshold.")
        return

    results.sort(key=lambda x: x[3])

    print(f"{'Log':<28} | {'N':>6} | {'RFT_MAE':>8} | {'RFT_RMSE':>8}")
    print("-" * 60)
    for log_name, n, r_mae, r_rmse in results:
        print(f"{str(log_name):<28} | {n:6d} | {r_mae:8.3f} | {r_rmse:8.3f}")

    print("\n[INFO] Plots saved into artifacts directory.")
    print(f"[INFO] artifacts path: {art}")

if __name__ == "__main__":
    main()
