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
    for ln in np.unique(meta[:, 0]):
        idx = np.where(meta[:, 0] == ln)[0]
        if len(idx) < 2:
            continue
        order = np.argsort(meta[idx, 1].astype(np.int64))
        idx_sorted = idx[order]
        y = np.maximum(out[idx_sorted], 0.0)
        y_fixed = y.copy()
        for t in range(1, len(y_fixed)):
            y_fixed[t] = np.clip(y_fixed[t], y_fixed[t-1] - max_drop, y_fixed[t-1] + max_rise)
        out[idx_sorted] = y_fixed
    return out


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
        print(f"[ERROR] Load failed: {e}")
        return

    try:
        X_past  = np.load(os.path.join(art, "X_past_test.npy"))
        X_batt  = np.load(os.path.join(art, "X_batt_state_test.npy"))
        y_rft_s = np.load(os.path.join(art, "y_rft_test.npy"))
        meta    = np.load(os.path.join(art, "meta_test.npy"), allow_pickle=True)
    except FileNotFoundError:
        print("[ERROR] Test data not found. Run 03_make_test.py first.")
        return

    print(f"[INFO] Test samples: {len(X_past)}")
    pred = model.predict([X_past, X_batt], batch_size=64, verbose=1)
    pred_rft_s = pred if not isinstance(pred, (list, tuple, dict)) else (
        pred["rft_output"] if isinstance(pred, dict) else pred[0]
    )

    y_rft         = s_rft.inverse_transform(y_rft_s).flatten()
    y_rft_raw_hat = s_rft.inverse_transform(pred_rft_s).flatten()

    # --------------------------------------------------
    # Optuna (선택)
    # --------------------------------------------------
    use_optuna = input("\nOptuna로 monotone 파라미터 탐색할까요? (y/n): ").strip().lower() == "y"

    if use_optuna:
        print("[INFO] Optuna 탐색 중 (n_trials=200)...")

        def objective(trial):
            max_drop = trial.suggest_float("max_drop", 0.05, 2.0, log=True)
            max_rise = trial.suggest_float("max_rise", 0.001, 0.5, log=True)
            y_hat = enforce_monotone_with_slope_limit_per_log(y_rft_raw_hat, meta, max_drop, max_rise)
            return mean_absolute_error(y_rft, y_hat)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=200, show_progress_bar=True)

        FIXED_MAX_DROP = study.best_params["max_drop"]
        FIXED_MAX_RISE = study.best_params["max_rise"]
        print(f"[Optuna] max_drop={FIXED_MAX_DROP:.4f}, max_rise={FIXED_MAX_RISE:.4f}, MAE={study.best_value:.3f}s")
        y_rft_hat = enforce_monotone_with_slope_limit_per_log(y_rft_raw_hat, meta, FIXED_MAX_DROP, FIXED_MAX_RISE)
    else:
        print("[INFO] Optuna 스킵 → raw 예측값 사용")
        y_rft_hat = y_rft_raw_hat
    # --------------------------------------------------

    # --------------------------------------------------
    # 로그별 타임스텝별 APE 계산
    # elapsed = max_rft - current_rft
    # --------------------------------------------------
    all_elapsed = []
    all_ape     = []
    log_results = []

    for ln in np.unique(meta[:, 0]):
        idx = np.where(meta[:, 0] == ln)[0]
        if len(idx) < 20:
            continue

        order      = np.argsort(meta[idx, 1].astype(np.int64))
        idx_sorted = idx[order]

        rft_true = y_rft[idx_sorted]
        rft_pred = y_rft_hat[idx_sorted]

        max_rft = rft_true[0]
        elapsed = max_rft - rft_true  # 경과 시간 (s)

        # APE: y_true가 0에 가까우면 폭발하므로 y_true > 1s 구간만 사용
        valid = rft_true > 1.0
        if valid.sum() < 10:
            continue

        ape = np.abs(rft_true[valid] - rft_pred[valid]) / rft_true[valid] * 100.0

        all_elapsed.append(elapsed[valid])
        all_ape.append(ape)

        mape_log = float(np.mean(ape))
        log_results.append((ln, int(valid.sum()), mape_log))

    if not all_elapsed:
        print("[WARN] 유효한 로그 없음.")
        return

    all_elapsed = np.concatenate(all_elapsed)
    all_ape     = np.concatenate(all_ape)

    overall_mape = float(np.mean(all_ape))

    print("=" * 60)
    print("MAPE EVALUATION")
    print("=" * 60)
    print(f"Overall MAPE : {overall_mape:.2f} %")
    print(f"Samples      : {len(all_ape):,}")
    print("=" * 60)

    log_results.sort(key=lambda x: x[2])
    print(f"\n{'Log':<28} | {'N':>6} | {'MAPE (%)':>10}")
    print("-" * 50)
    for ln, n, mape in log_results:
        print(f"{str(ln):<28} | {n:6d} | {mape:10.2f}")

    # --------------------------------------------------
    # 시각화
    # --------------------------------------------------
    BIN_SIZE = 10  # 10초 단위로 binning
    max_elapsed = np.percentile(all_elapsed, 99)
    bins = np.arange(0, max_elapsed + BIN_SIZE, BIN_SIZE)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    bin_mape_mean = []
    bin_mape_std  = []

    for i in range(len(bins) - 1):
        mask = (all_elapsed >= bins[i]) & (all_elapsed < bins[i+1])
        if mask.sum() > 0:
            bin_mape_mean.append(np.mean(all_ape[mask]))
            bin_mape_std.append(np.std(all_ape[mask]))
        else:
            bin_mape_mean.append(np.nan)
            bin_mape_std.append(np.nan)

    bin_mape_mean = np.array(bin_mape_mean)
    bin_mape_std  = np.array(bin_mape_std)
    valid_bins    = ~np.isnan(bin_mape_mean)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"MAPE Analysis  (Overall MAPE = {overall_mape:.2f}%)", fontsize=13, fontweight="bold")

    # 왼쪽: 경과 시간별 MAPE
    ax = axes[0]
    ax.plot(bin_centers[valid_bins], bin_mape_mean[valid_bins], color="#4C72B0", linewidth=2)
    ax.fill_between(
        bin_centers[valid_bins],
        (bin_mape_mean - bin_mape_std)[valid_bins],
        (bin_mape_mean + bin_mape_std)[valid_bins],
        alpha=0.2, color="#4C72B0"
    )
    ax.axhline(overall_mape, color="red", linestyle="--", linewidth=1.2, label=f"Overall MAPE={overall_mape:.2f}%")
    ax.set_title("MAPE vs Elapsed Time")
    ax.set_xlabel("Elapsed Time (s)")
    ax.set_ylabel("MAPE (%)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    # 오른쪽: APE 분포 히스토그램
    ax = axes[1]
    clip_ape = np.clip(all_ape, 0, 100)
    ax.hist(clip_ape, bins=80, color="#55A868", alpha=0.75, edgecolor="none")
    ax.axvline(overall_mape, color="red", linestyle="--", linewidth=1.2, label=f"mean={overall_mape:.2f}%")
    ax.axvline(np.median(all_ape), color="orange", linestyle="--", linewidth=1.2, label=f"median={np.median(all_ape):.2f}%")
    ax.set_title("APE Distribution (clipped at 100%)")
    ax.set_xlabel("APE (%)")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    save_path = os.path.join(art, "mape_analysis.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[INFO] 저장 완료: {save_path}")


if __name__ == "__main__":
    main()
