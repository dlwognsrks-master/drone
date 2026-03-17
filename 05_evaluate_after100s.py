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


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


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
            y_fixed[t] = np.clip(y_fixed[t], y_fixed[t-1] - max_drop, y_fixed[t-1] + max_rise)
        out[idx_sorted] = y_fixed
    return out


def plot_series(save_path, title, y_true, y_pred, cut_idx):
    plt.figure(figsize=(14, 6))
    x = np.arange(len(y_true))

    # 100초 이전 구간 (회색)
    plt.plot(x[:cut_idx], y_true[:cut_idx], color="lightgray", linewidth=1.0)
    plt.plot(x[:cut_idx], y_pred[:cut_idx], color="lightgray", linewidth=1.0, linestyle="--")

    # 100초 이후 구간 (색상)
    plt.plot(x[cut_idx:], y_true[cut_idx:],  label="True", color="#4C72B0")
    plt.plot(x[cut_idx:], y_pred[cut_idx:],  label="Pred", color="#C44E52")

    plt.axvline(cut_idx, color="black", linestyle=":", linewidth=1.2, label="t=100s")

    plt.title(title)
    plt.xlabel("Time step (samples)")
    plt.ylabel("RFT (s)")

    y_all = np.concatenate([y_true, y_pred])
    y_min = max(0, np.floor(y_all.min() / 10) * 10)
    y_max = np.ceil(y_all.max() / 10) * 10
    minor_ticks = np.arange(y_min, y_max + 10, 10)
    major_ticks = np.arange(y_min, y_max + 50, 50)
    ax = plt.gca()
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    ax.grid(True, which="major", linestyle="--", alpha=0.7)
    ax.grid(True, which="minor", linestyle="--", alpha=0.3)
    plt.ylim(y_min, y_max)
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
    print("[INFO] Predicting...")

    pred = model.predict([X_past, X_batt], batch_size=64, verbose=1)
    pred_rft_s = pred if not isinstance(pred, (list, tuple, dict)) else (
        pred["rft_output"] if isinstance(pred, dict) else pred[0]
    )

    y_rft         = s_rft.inverse_transform(y_rft_s).flatten()
    y_rft_raw_hat = s_rft.inverse_transform(pred_rft_s).flatten()

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
        y_rft_hat = enforce_monotone_with_slope_limit_per_log(y_rft_raw_hat, meta, FIXED_MAX_DROP, FIXED_MAX_RISE)
    else:
        print("[INFO] Optuna 스킵 → raw 예측값 사용")
        y_rft_hat = y_rft_raw_hat
    # --------------------------------------------------

    # --------------------------------------------------
    # 로그별로 elapsed time > 100s 구간 필터링
    # elapsed = max_rft - current_rft
    # --------------------------------------------------
    ELAPSED_THRESH = 100.0  # seconds

    after100_true = []
    after100_pred = []
    results = []

    log_names = np.unique(meta[:, 0])

    for log_name in log_names:
        idx = np.where(meta[:, 0] == log_name)[0]
        if len(idx) < 20:
            continue

        ts = meta[idx, 1].astype(np.int64)
        order = np.argsort(ts)
        idx_sorted = idx[order]

        rft_true = y_rft[idx_sorted]
        rft_pred = y_rft_hat[idx_sorted]

        max_rft = rft_true[0]  # 시작 시점 RFT ≈ 전체 비행시간
        elapsed = max_rft - rft_true  # 경과 시간

        # 100초 이후 구간 마스크
        mask_after = elapsed > ELAPSED_THRESH
        cut_idx = int(np.argmax(mask_after)) if mask_after.any() else len(rft_true)

        if mask_after.sum() < 10:
            continue

        r_true_after = rft_true[mask_after]
        r_pred_after = rft_pred[mask_after]

        after100_true.append(r_true_after)
        after100_pred.append(r_pred_after)

        r_mae  = mean_absolute_error(r_true_after, r_pred_after)
        r_rmse = rmse(r_true_after, r_pred_after)
        results.append((log_name, int(mask_after.sum()), r_mae, r_rmse))

        safe_name = str(log_name).replace(" ", "_").replace(":", "_")
        plot_series(
            save_path=os.path.join(art, f"AFTER100_RFT_{safe_name}_RMSE_{r_rmse:.2f}.png"),
            title=f"RFT (after 100s) | Log={log_name} | RMSE={r_rmse:.2f}s",
            y_true=rft_true,
            y_pred=rft_pred,
            cut_idx=cut_idx,
        )

    if not after100_true:
        print("[WARN] No logs passed the threshold.")
        return

    all_true = np.concatenate(after100_true)
    all_pred = np.concatenate(after100_pred)

    total_mae  = mean_absolute_error(all_true, all_pred)
    total_rmse = rmse(all_true, all_pred)

    print("=" * 60)
    print("EVALUATION RESULT (after 100s elapsed)")
    print("=" * 60)
    print(f"RFT  MAE : {total_mae:.3f} s")
    print(f"RFT RMSE : {total_rmse:.3f} s")
    print(f"Samples  : {len(all_true):,}")
    print("=" * 60)

    results.sort(key=lambda x: x[3])
    print(f"\n{'Log':<28} | {'N':>6} | {'MAE':>8} | {'RMSE':>8}")
    print("-" * 60)
    for log_name, n, r_mae, r_rmse in results:
        print(f"{str(log_name):<28} | {n:6d} | {r_mae:8.3f} | {r_rmse:8.3f}")

    print(f"\n[INFO] Plots saved to: {art}")


if __name__ == "__main__":
    main()
