import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import config
from utils import find_csv_file, _dedup_keep_last, zoh_resample


def load_velocity_altitude(log_folder):
    lp_path = find_csv_file(log_folder, "*vehicle_local_position_0.csv")
    vs_path = find_csv_file(log_folder, "*vehicle_status_0.csv")
    bt_path = find_csv_file(log_folder, "*battery_status_0.csv")

    if not lp_path or not vs_path or not bt_path:
        return None

    try:
        df_lp = pd.read_csv(lp_path).sort_values("timestamp").reset_index(drop=True)
        df_vs = pd.read_csv(vs_path).sort_values("timestamp").reset_index(drop=True)
        df_bt = pd.read_csv(bt_path).sort_values("timestamp").reset_index(drop=True)

        needed = ["timestamp", "vx", "vy", "vz", "z"]
        if not all(c in df_lp.columns for c in needed):
            return None

        ts = df_lp["timestamp"].to_numpy(dtype=np.int64)
        ts = ts[np.isfinite(ts)].astype(np.int64)
        if ts.size < 10:
            return None
        ts, _ = _dedup_keep_last(ts, np.arange(ts.size))

        # failsafe cut
        nav_state = zoh_resample(ts, df_vs["timestamp"].to_numpy(dtype=np.int64), df_vs["nav_state"].to_numpy())
        batt_warn = zoh_resample(ts, df_bt["timestamp"].to_numpy(dtype=np.int64), df_bt["warning"].to_numpy())
        cut = np.where((nav_state == 18) & (batt_warn >= 2))[0]
        if cut.size > 0:
            ts = ts[:int(cut[0])]
        if ts.size < 10:
            return None

        lp_ts = df_lp["timestamp"].to_numpy(dtype=np.int64)
        vx = zoh_resample(ts, lp_ts, df_lp["vx"].to_numpy()).astype(np.float32)
        vy = zoh_resample(ts, lp_ts, df_lp["vy"].to_numpy()).astype(np.float32)
        vz = zoh_resample(ts, lp_ts, df_lp["vz"].to_numpy()).astype(np.float32)
        z  = zoh_resample(ts, lp_ts, df_lp["z"].to_numpy()).astype(np.float32)

        # NED → 고도(양수)
        altitude = -z

        # 고도 변화율 (m/s), timestamp microsecond → second
        dt = np.diff(ts.astype(np.float64)) / 1e6
        dz_dt = np.diff(altitude.astype(np.float64)) / np.where(dt > 0, dt, 1e-6)

        return {
            "vx": vx, "vy": vy, "vz": vz,
            "altitude": altitude,
            "dz_dt": dz_dt.astype(np.float32),
        }

    except Exception as e:
        print(f"[WARN] {os.path.basename(log_folder)}: {e}")
        return None


def gather_all_logs():
    roots = [config.DATA_ROOT_DIR, config.TEST_LOG_ROOT]
    folders = []
    for root in roots:
        folders += [
            f for f in glob.glob(os.path.join(root, "**", "log_*"), recursive=True)
            if os.path.isdir(f)
        ]
    return folders


def main():
    os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)

    folders = gather_all_logs()
    print(f"[INFO] 총 로그 수: {len(folders)}")

    all_vx, all_vy, all_vz = [], [], []
    all_alt, all_dzdt = [], []

    for folder in folders:
        result = load_velocity_altitude(folder)
        if result is None:
            continue
        all_vx.append(result["vx"])
        all_vy.append(result["vy"])
        all_vz.append(result["vz"])
        all_alt.append(result["altitude"])
        all_dzdt.append(result["dz_dt"])

    if not all_vx:
        print("[ERROR] 데이터 없음.")
        return

    vx   = np.concatenate(all_vx)
    vy   = np.concatenate(all_vy)
    vz   = np.concatenate(all_vz)
    alt  = np.concatenate(all_alt)
    dzdt = np.concatenate(all_dzdt)

    h_speed = np.sqrt(vx**2 + vy**2)
    total_speed = np.sqrt(vx**2 + vy**2 + vz**2)

    print(f"[INFO] 총 샘플 수: {len(vx):,}")

    # ── 시각화 ──────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("Drone Data Distribution", fontsize=16, fontweight="bold")
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    def hist(ax, data, title, xlabel, color, clip=None):
        if clip:
            data = np.clip(data, clip[0], clip[1])
        ax.hist(data, bins=100, color=color, alpha=0.75, edgecolor="none")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.axvline(np.mean(data), color="red", linewidth=1.2, linestyle="--", label=f"mean={np.mean(data):.2f}")
        ax.legend(fontsize=8)

    hist(fig.add_subplot(gs[0, 0]), vx,          "vx Distribution",           "vx (m/s)",    "#4C72B0")
    hist(fig.add_subplot(gs[0, 1]), vy,          "vy Distribution",           "vy (m/s)",    "#4C72B0")
    hist(fig.add_subplot(gs[0, 2]), vz,          "vz Distribution",           "vz (m/s)",    "#4C72B0")
    hist(fig.add_subplot(gs[1, 0]), h_speed,     "Horizontal Speed",          "speed (m/s)", "#55A868")
    hist(fig.add_subplot(gs[1, 1]), total_speed, "Total Speed",               "speed (m/s)", "#55A868")
    hist(fig.add_subplot(gs[1, 2]), alt,         "Altitude Distribution",     "altitude (m)","#C44E52")
    hist(fig.add_subplot(gs[2, 0]), dzdt,        "Altitude Change Rate",      "dz/dt (m/s)", "#8172B2", clip=(-10, 10))

    # 통계 텍스트
    ax_stat = fig.add_subplot(gs[2, 1:])
    ax_stat.axis("off")
    stats = (
        f"{'Feature':<18} {'mean':>8} {'std':>8} {'min':>8} {'max':>8}\n"
        f"{'-'*54}\n"
        f"{'vx (m/s)':<18} {np.mean(vx):>8.2f} {np.std(vx):>8.2f} {np.min(vx):>8.2f} {np.max(vx):>8.2f}\n"
        f"{'vy (m/s)':<18} {np.mean(vy):>8.2f} {np.std(vy):>8.2f} {np.min(vy):>8.2f} {np.max(vy):>8.2f}\n"
        f"{'vz (m/s)':<18} {np.mean(vz):>8.2f} {np.std(vz):>8.2f} {np.min(vz):>8.2f} {np.max(vz):>8.2f}\n"
        f"{'h_speed (m/s)':<18} {np.mean(h_speed):>8.2f} {np.std(h_speed):>8.2f} {np.min(h_speed):>8.2f} {np.max(h_speed):>8.2f}\n"
        f"{'altitude (m)':<18} {np.mean(alt):>8.2f} {np.std(alt):>8.2f} {np.min(alt):>8.2f} {np.max(alt):>8.2f}\n"
        f"{'dz/dt (m/s)':<18} {np.mean(dzdt):>8.2f} {np.std(dzdt):>8.2f} {np.min(dzdt):>8.2f} {np.max(dzdt):>8.2f}\n"
    )
    ax_stat.text(0.02, 0.95, stats, transform=ax_stat.transAxes,
                 fontsize=9, verticalalignment="top", fontfamily="monospace",
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.4))

    save_path = os.path.join(config.ARTIFACTS_DIR, "data_distribution.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] 저장 완료: {save_path}")


if __name__ == "__main__":
    main()
