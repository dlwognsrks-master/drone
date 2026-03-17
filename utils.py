import os
import glob
import numpy as np
import pandas as pd
import config



#------------------------------------------------------------------------------------#
# [1] 데이터 전처리 (시계열 데이터 전처리)
# csv 파일 찾기 - 쿼터니언 각도 오일러 변환 - _dedup_keep_last - _ffill_nan_1d - zoh_resample
#------------------------------------------------------------------------------------#

def find_csv_file(log_folder, file_pattern):
    search_path = os.path.join(log_folder, "**", file_pattern)
    files = glob.glob(search_path, recursive=True)
   
    if not files:
        return None
    
    # [Validation] 파일이 여러 개 발견될 경우 경고 또는 예외 발생
    if len(files) > 1:
        # 가장 최근에 수정된 파일을 선택하거나, 특정 경로(예: 'csv/')가 포함된 것 우선
        print(f"[WARNING] Multiple files found for {file_pattern} in {log_folder}")
        # 예: 경로에 'csv/'가 포함된 파일만 필터링하거나, 알파벳순 정렬 후 선택
        files.sort() 
        
    return files[0]


def transform_quat_to_euler(w, x, y, z):
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = np.where(np.abs(sinp) >= 1, np.sign(sinp) * (np.pi / 2), np.arcsin(sinp))

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


# 같은 시간에 기록된 데이터가 여러 개 있을 때, 가장 마지막 값만 남기고 나머지는 모두 지움
def _dedup_keep_last(src_ts: np.ndarray, src_data: np.ndarray):
    src_ts = np.asarray(src_ts, dtype=np.int64)
    src_data = np.asarray(src_data)

    if src_ts.size == 0:
        return src_ts, src_data

    order = np.argsort(src_ts, kind="mergesort")
    ts_sorted = src_ts[order]
    data_sorted = src_data[order]

    last_mask = np.ones_like(ts_sorted, dtype=bool)
    last_mask[:-1] = ts_sorted[:-1] != ts_sorted[1:]       
    return ts_sorted[last_mask], data_sorted[last_mask]  # 동일한 timestamp에서 들어온 중복 데이터를 제거한 정제된 데이터 반환



# ZOH ( Zero Order Hold) : 제어 공학기법 / 새로운 데이터가 들어오기 전까지 직전의 값을 그대로 유지한다.
# Pandas의 ffill() 과 비슷한 로직. 근데 Pandas 는 무거운 라이브러리. 난 Numpy 만 사용하여 메모리와 속도 좋게함.
def _ffill_nan_1d(x: np.ndarray):
    x = np.asarray(x)
    # 1차원 리스트가 아니면 에러를 내는 안정장치
    if x.ndim != 1:
        raise ValueError("_ffill_nan_1d expects 1D array")
    if x.size == 0:
        return x

    out = x.copy()
    valid = np.isfinite(out)   # True/False 변환
    if not np.any(valid):
        return np.zeros_like(out)

    first_valid_idx = int(np.argmax(valid))
    out[:first_valid_idx] = out[first_valid_idx]

    for i in range(first_valid_idx + 1, out.size):
        if not np.isfinite(out[i]):
            out[i] = out[i - 1]
    return out



# 서로 다른 주기로 들어오는 센서 데이터를 하나의 타임라인으로 정렬하는 데이터 전처리 최종단계
def zoh_resample(target_ts, src_ts, src_data):
    target_ts = np.asarray(target_ts, dtype=np.int64)
    src_ts = np.asarray(src_ts, dtype=np.int64)
    src_data = np.asarray(src_data)

    if src_ts.size == 0 or src_data.size == 0:
        raise ValueError("zoh_resample: empty src")
    if src_ts.shape[0] != src_data.shape[0]:
        raise ValueError("zoh_resample: length mismatch")

    # 중복 데이터 제거하고 최신 값만 남김
    ts_uniq, data_uniq = _dedup_keep_last(src_ts, src_data)

    # 데이터에서 NaN을 ZOH 로직으로 채워넣음
    if np.issubdtype(data_uniq.dtype, np.number) and data_uniq.ndim == 1:
        data_uniq = _ffill_nan_1d(data_uniq)


    # ZOH 로직 완성
    # 단순 반복문(for-loop)을 통한 데이터 검색보다 수만 배 이상 빠른 성능을 제공
    # 드론과 같이 대용량 시계열 데이터지만, 실시간성도 요구하는 시스템에서 최적화
    idx = np.searchsorted(ts_uniq, target_ts, side="right") - 1
    idx = np.clip(idx, 0, len(ts_uniq) - 1)
    return data_uniq[idx]


# ✅ [추가] 고주파 진동 성분을 평탄화(Smoothing)한 뒤 리샘플링하는 EMA 필터 함수
def time_aware_ema_resample(target_ts, src_ts, src_data, cutoff_freq=2.0):
    target_ts = np.asarray(target_ts, dtype=np.int64)
    src_ts = np.asarray(src_ts, dtype=np.int64)
    src_data = np.asarray(src_data, dtype=np.float32)

    if src_ts.size < 2:
        return np.zeros_like(target_ts, dtype=np.float32)

    filtered_data = np.empty_like(src_data)
    filtered_data[0] = src_data[0]
    
    # 시간 간격(dt) 계산 (microsecond -> second)
    src_ts_sec = src_ts.astype(np.float64) / 1e6
    dts = np.diff(src_ts_sec, prepend=src_ts_sec[0])

    for i in range(1, len(src_data)):
        dt = dts[i] if dts[i] > 0 else 1e-6
        # alpha = dt / (tau + dt), tau = 1 / (2 * pi * f_c)
        tau = 1.0 / (2.0 * np.pi * cutoff_freq)
        alpha = dt / (tau + dt)
        alpha = np.clip(alpha, 0.0, 1.0)
        filtered_data[i] = alpha * src_data[i] + (1.0 - alpha) * filtered_data[i-1]

    ts_uniq, data_uniq = _dedup_keep_last(src_ts, filtered_data)
    idx = np.searchsorted(ts_uniq, target_ts, side="right") - 1
    idx = np.clip(idx, 0, len(ts_uniq) - 1)
    return data_uniq[idx]




#------------------------------------------------------------------------------------#
# [2] 각기 다른 센서와 제어 모듈에서 나오는 개별 csv파일들을 하나의 데이터셋으로 통합
#------------------------------------------------------------------------------------#

 
def load_and_process_log(log_folder):
    log_name = os.path.basename(log_folder)

    # 어떤 이름으로 어떤 파일을 찾을 것인가
    files = {
        "traj_sp": "*trajectory_setpoint_0.csv",
        "vehicle_att_sp": "*vehicle_attitude_setpoint_0.csv",
        "vehicle_local_position": "*vehicle_local_position_0.csv",
        "battery": "*battery_status_0.csv",
        "vehicle_status": "*vehicle_status_0.csv",

        # ✅ 추가
        "actuator_motors": "*actuator_motors_0.csv",
        "vehicle_angular_velocity": "*vehicle_angular_velocity_0.csv",
        "vehicle_attitude": "*vehicle_attitude_0.csv",
    }

    paths = {k: find_csv_file(log_folder, v) for k, v in files.items()}

    # 파일 자체가 없으면 학습 제외
    if not all(paths.values()):
        return []

    # 파일 읽기 - 시간순 정렬 - 인덱스 초기화
    try:
        df_traj_sp = pd.read_csv(paths["traj_sp"]).sort_values("timestamp").reset_index(drop=True)
        df_vehicle_att_sp = pd.read_csv(paths["vehicle_att_sp"]).sort_values("timestamp").reset_index(drop=True)
        df_vehicle_local_position = pd.read_csv(paths["vehicle_local_position"]).sort_values("timestamp").reset_index(drop=True)
        df_battery = pd.read_csv(paths["battery"]).sort_values("timestamp").reset_index(drop=True)
        df_vehicle_status = pd.read_csv(paths["vehicle_status"]).sort_values("timestamp").reset_index(drop=True)

        # ✅ 추가
        df_actuator_motors = pd.read_csv(paths["actuator_motors"]).sort_values("timestamp").reset_index(drop=True)
        df_vehicle_ang_vel = pd.read_csv(paths["vehicle_angular_velocity"]).sort_values("timestamp").reset_index(drop=True)
        df_vehicle_att = pd.read_csv(paths["vehicle_attitude"]).sort_values("timestamp").reset_index(drop=True)

        # ------------------- #
        # 기준 timestamp 맞추기 (traj_sp timestamp 기준)
        # ------------------- #
        ts = df_traj_sp["timestamp"].to_numpy(dtype=np.int64)
        ts = ts[np.isfinite(ts)].astype(np.int64, copy=False)
        if ts.size < 10:
            return []

        # ts 중복 제거(마지막 값 유지)
        ts, _ = _dedup_keep_last(ts, np.arange(ts.size))
        if ts.size < 10:
            return []

        # ------------------- #
        # failsafe cut (ZOH)
        # ------------------- #
        nav_state = zoh_resample(
            ts,
            df_vehicle_status["timestamp"].to_numpy(dtype=np.int64),
            df_vehicle_status["nav_state"].to_numpy(),
        )
        batt_warning = zoh_resample(
            ts,
            df_battery["timestamp"].to_numpy(dtype=np.int64),
            df_battery["warning"].to_numpy(),
        )

        failsafe_idx = np.where((nav_state == 18) & (batt_warning >= 2))[0]
        if failsafe_idx.size > 0:
            ts = ts[: int(failsafe_idx[0])]
        if ts.size < 10:
            return []

        end_time = int(ts[-1])
        N = int(ts.size)

        # ------------------- #
        # 데이터 최소 길이 체크(현재 포함 정렬에 맞게)
        # ------------------- #
        min_need = max(
            (config.PAST_SEQ_LEN - 1) * config.PAST_STRIDE,
            (2 * config.DELTA_I_H) - 1,
            config.SP_SEQ_LEN - 1,
        ) + 1

        if N < min_need:
            return []

        # ------------------------------------ #
        # 2) Resample signals to ts using ZOH
        # ------------------------------------ #

        # traj setpoint vx/vy/vz (⚠️ AE 입력에서는 제거하지만 x_sp(평균전류용) 때문에 유지)
        if "vx" in df_traj_sp.columns:
            vx_src = df_traj_sp["vx"].to_numpy()
            vy_src = df_traj_sp["vy"].to_numpy()
            vz_src = df_traj_sp["vz"].to_numpy()
        else:
            vx_src = df_traj_sp["velocity[0]"].to_numpy()
            vy_src = df_traj_sp["velocity[1]"].to_numpy()
            vz_src = df_traj_sp["velocity[2]"].to_numpy()

        traj_ts = df_traj_sp["timestamp"].to_numpy(dtype=np.int64)
        traj_sp_vx = zoh_resample(ts, traj_ts, vx_src).astype(np.float32)
        traj_sp_vy = zoh_resample(ts, traj_ts, vy_src).astype(np.float32)
        traj_sp_vz = zoh_resample(ts, traj_ts, vz_src).astype(np.float32)

        # thrust setpoint
        possible_cols = ["thrust_body.02", "thrust_body[2]", "thrust_body"]
        thrust_col = next((c for c in possible_cols if c in df_vehicle_att_sp.columns), None)
        if thrust_col is None:
            return []

        thrust_src = np.abs(df_vehicle_att_sp[thrust_col].to_numpy())
        thrust_sp = zoh_resample(
            ts,
            df_vehicle_att_sp["timestamp"].to_numpy(dtype=np.int64),
            thrust_src
        ).astype(np.float32)

        # local velocity + acceleration
        needed_lp = ["timestamp", "vx", "vy", "vz", "ax", "ay", "az"]
        if not all(c in df_vehicle_local_position.columns for c in needed_lp):
            return []

        local_ts = df_vehicle_local_position["timestamp"].to_numpy(dtype=np.int64)
        local_vx = zoh_resample(ts, local_ts, df_vehicle_local_position["vx"].to_numpy()).astype(np.float32)
        local_vy = zoh_resample(ts, local_ts, df_vehicle_local_position["vy"].to_numpy()).astype(np.float32)
        local_vz = zoh_resample(ts, local_ts, df_vehicle_local_position["vz"].to_numpy()).astype(np.float32)
        
        # ✅ 가속도 데이터: 고주파 진동 억제를 위해 EMA 필터 적용
        local_ax = time_aware_ema_resample(ts, local_ts, df_vehicle_local_position["ax"].to_numpy(), cutoff_freq=2.0)
        local_ay = time_aware_ema_resample(ts, local_ts, df_vehicle_local_position["ay"].to_numpy(), cutoff_freq=2.0)
        local_az = time_aware_ema_resample(ts, local_ts, df_vehicle_local_position["az"].to_numpy(), cutoff_freq=2.0)

        # vehicle angular velocity + angular acceleration
        needed_ang = [
            "timestamp",
            "xyz[0]", "xyz[1]", "xyz[2]",
            "xyz_derivative[0]", "xyz_derivative[1]", "xyz_derivative[2]",
        ]
        if not all(c in df_vehicle_ang_vel.columns for c in needed_ang):
            return []
        
        ang_ts = df_vehicle_ang_vel["timestamp"].to_numpy(dtype=np.int64)
        
        # Angular Velocity (✅ 여기도 EMA 필터 적용하여 평탄화)
        ang_p = time_aware_ema_resample(ts, ang_ts, df_vehicle_ang_vel["xyz[0]"].to_numpy(), cutoff_freq=2.0)
        ang_q = time_aware_ema_resample(ts, ang_ts, df_vehicle_ang_vel["xyz[1]"].to_numpy(), cutoff_freq=2.0)
        ang_r = time_aware_ema_resample(ts, ang_ts, df_vehicle_ang_vel["xyz[2]"].to_numpy(), cutoff_freq=2.0)

        # ✅ 각가속도 데이터: 고주파 진동 억제를 위해 EMA 필터 적용
        ang_dp = time_aware_ema_resample(ts, ang_ts, df_vehicle_ang_vel["xyz_derivative[0]"].to_numpy(), cutoff_freq=2.0)
        ang_dq = time_aware_ema_resample(ts, ang_ts, df_vehicle_ang_vel["xyz_derivative[1]"].to_numpy(), cutoff_freq=2.0)
        ang_dr = time_aware_ema_resample(ts, ang_ts, df_vehicle_ang_vel["xyz_derivative[2]"].to_numpy(), cutoff_freq=2.0)

        
        # vehicle attitude (quaternion → roll, pitch)
        needed_att = ["timestamp", "q[0]", "q[1]", "q[2]", "q[3]"]
        if not all(c in df_vehicle_att.columns for c in needed_att):
            return []

        att_ts = df_vehicle_att["timestamp"].to_numpy(dtype=np.int64)
        q0 = zoh_resample(ts, att_ts, df_vehicle_att["q[0]"].to_numpy()).astype(np.float64)
        q1 = zoh_resample(ts, att_ts, df_vehicle_att["q[1]"].to_numpy()).astype(np.float64)
        q2 = zoh_resample(ts, att_ts, df_vehicle_att["q[2]"].to_numpy()).astype(np.float64)
        q3 = zoh_resample(ts, att_ts, df_vehicle_att["q[3]"].to_numpy()).astype(np.float64)
        roll, pitch, _ = transform_quat_to_euler(q0, q1, q2, q3)
        roll  = roll.astype(np.float32)
        pitch = pitch.astype(np.float32)

        # actuator motors (control[0]~[3]) 0~1
        needed_mot = ["timestamp", "control[0]", "control[1]", "control[2]", "control[3]"]
        if not all(c in df_actuator_motors.columns for c in needed_mot):
            return []

        mot_ts = df_actuator_motors["timestamp"].to_numpy(dtype=np.int64)
        mot0 = zoh_resample(ts, mot_ts, df_actuator_motors["control[0]"].to_numpy()).astype(np.float32)
        mot1 = zoh_resample(ts, mot_ts, df_actuator_motors["control[1]"].to_numpy()).astype(np.float32)
        mot2 = zoh_resample(ts, mot_ts, df_actuator_motors["control[2]"].to_numpy()).astype(np.float32)
        mot3 = zoh_resample(ts, mot_ts, df_actuator_motors["control[3]"].to_numpy()).astype(np.float32)

        # battery
        if not all(c in df_battery.columns for c in ["timestamp", "voltage_v", "current_a", "remaining", "warning", "discharged_mah"]):
            return []

        batt_ts = df_battery["timestamp"].to_numpy(dtype=np.int64)
        voltage_v = zoh_resample(ts, batt_ts, df_battery["voltage_v"].to_numpy()).astype(np.float32)
        current_a = zoh_resample(ts, batt_ts, df_battery["current_a"].to_numpy()).astype(np.float32)
        remaining = zoh_resample(ts, batt_ts, df_battery["remaining"].to_numpy()).astype(np.float32)

        # consumed mAh
        consumed_mah = zoh_resample(ts, batt_ts, df_battery["discharged_mah"].to_numpy()).astype(np.float32)

        # RFT (offline label)
        rft_s = ((end_time - ts).astype(np.float64) / 1e6).astype(np.float32)

        # ============================================================
        # flight_features (past input): 21 dims
        # 0..2  : local_v (vx, vy, vz)
        # 3..5  : ang_vel (p, q, r)
        # 6..8  : ang_acc (dp/dt, dq/dt, dr/dt)
        # 9     : thrust_sp
        # 10..13: motors 0..3
        # 14    : voltage
        # 15    : current
        # 16    : roll
        # 17    : pitch
        # 18..20: traj_sp (vx_sp, vy_sp, vz_sp)
        # ============================================================
        flight_features = np.column_stack(
            [
                local_vx, local_vy, local_vz,
                ang_p, ang_q, ang_r,
                ang_dp, ang_dq, ang_dr,
                thrust_sp,
                mot0, mot1, mot2, mot3,
                voltage_v,
                current_a,
                roll, pitch,
                traj_sp_vx, traj_sp_vy, traj_sp_vz,
            ]
        ).astype(np.float32)

        # 차원 체크 (강추)
        if flight_features.shape[1] != 21:
            print(f"[ERROR] flight_features dim mismatch: {flight_features.shape[1]} != 21")
            return []

        # ==============================
        # EMA helpers (battery_state)
        # ==============================
        def ema_1d(x: np.ndarray, alpha: float) -> np.ndarray:
            y = np.empty_like(x, dtype=np.float32)
            y[0] = x[0]
            for i in range(1, len(x)):
                y[i] = alpha * y[i - 1] + (1.0 - alpha) * x[i]
            return y

        hz = 5.0
        dt = 1.0 / hz

        half_life = 1.5
        alpha = float(np.exp(-np.log(2.0) / (half_life * hz)))

        V_now = voltage_v.astype(np.float32)
        I_now = current_a.astype(np.float32)

        V_ema = ema_1d(V_now, alpha)
        I_ema = ema_1d(I_now, alpha)

        sag = (V_now - V_ema).astype(np.float32)

        # battery_state (input_state): consumed_mah, V_ema, I_ema, sag
        battery_state = np.column_stack([
            consumed_mah.astype(np.float32),
            V_ema,
            I_ema,
            sag,
        ]).astype(np.float32)

        # setpoint stream for i_mean head (그대로 유지) 이것도 뭔가 손봐야할듯
        sp_stream = np.column_stack([traj_sp_vx, traj_sp_vy, traj_sp_vz, thrust_sp]).astype(np.float32)

        # ==============================
        # 4) Build samples (현재 포함 정렬)
        # ==============================
        samples = []

        STRIDE = int(config.PAST_STRIDE)
        PAST_LEN = int(config.PAST_SEQ_LEN)
        DELTA_I = int(config.DELTA_I_H)
        SP_LEN = int(config.SP_SEQ_LEN)

        start_i = max((PAST_LEN - 1) * STRIDE, DELTA_I - 1, SP_LEN - 1)

        for i in range(start_i, N):
            past_indices = i - np.arange(PAST_LEN - 1, -1, -1) * STRIDE

            x_past = flight_features[past_indices]  # (PAST_LEN, 19)
            x_sp = sp_stream[i - (SP_LEN - 1): i + 1]  # (SP_LEN, 4)
            x_batt_state = battery_state[i]  # (6,)

            y_rft = rft_s[i]  # scalar

            # 최근 1초 평균전류 (DELTA_I_H=5이면 5샘플=1초)
            i_now = current_a.astype(np.float32)
            i_mean = float(np.mean(i_now[i - (DELTA_I - 1): i + 1]))

            samples.append({
                "x_past": x_past,
                "x_sp": x_sp,
                "x_batt_state": x_batt_state,
                "y_rft": y_rft,
                "y_i_mean": i_mean,
                "meta": np.array([log_name, ts[i]], dtype=object),
            })

        return samples

    except Exception as e:
        print(f"[ERROR] load_and_process_log failed for {log_folder}: {e}")
        return []