import os
import glob
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GroupShuffleSplit

import config
import utils


class CustomMixedScaler:
    """
    일반 feature column에는 StandardScaler를 적용하고,
    0~1 사이로 물리적 한계가 명확한 column에는 MinMaxScaler를 적용하여
    극단치 발생을 막고 분산 스케일을 맞추는 래퍼(wrap) 스케일러.
    """

    def __init__(self, bounded_cols, feature_range=(-2.0, 2.0)):
        self.bounded_cols = sorted(set(int(i) for i in bounded_cols))
        self.feature_range = feature_range
        self._standard_scaler = StandardScaler()
        self._minmax_scaler = MinMaxScaler(feature_range=self.feature_range)
        self._scale_cols = None
        self._n_features_in_ = None

    def fit(self, X):
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"CustomMixedScaler.fit expects 2D array, got ndim={X.ndim}")
        
        n, f = X.shape
        self._n_features_in_ = int(f)
        all_cols = list(range(f))
        self._scale_cols = [c for c in all_cols if c not in self.bounded_cols]
        
        if len(self._scale_cols) > 0:
            self._standard_scaler.fit(X[:, self._scale_cols])
            
        if len(self.bounded_cols) > 0:
            self._minmax_scaler.fit(X[:, self.bounded_cols])
            
        return self

    def transform(self, X):
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"CustomMixedScaler.transform expects 2D array, got ndim={X.ndim}")
        
        n, f = X.shape
        if self._n_features_in_ is None:
            raise ValueError("Scaler is not fitted yet.")
        if f != self._n_features_in_:
            raise ValueError(f"Feature dim mismatch: got {f}, expected {self._n_features_in_}")

        out = X.astype(np.float32, copy=True)
        
        if len(self._scale_cols) > 0:
            out[:, self._scale_cols] = self._standard_scaler.transform(out[:, self._scale_cols]).astype(np.float32, copy=False)
            
        if len(self.bounded_cols) > 0:
            out[:, self.bounded_cols] = self._minmax_scaler.transform(out[:, self.bounded_cols]).astype(np.float32, copy=False)
            
        return out

    def inverse_transform(self, X):
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"CustomMixedScaler.inverse_transform expects 2D array, got ndim={X.ndim}")
        
        n, f = X.shape
        if self._n_features_in_ is None:
            raise ValueError("Scaler is not fitted yet.")
        if f != self._n_features_in_:
            raise ValueError(f"Feature dim mismatch: got {f}, expected {self._n_features_in_}")

        out = X.astype(np.float32, copy=True)
        
        if len(self._scale_cols) > 0:
            out[:, self._scale_cols] = self._standard_scaler.inverse_transform(out[:, self._scale_cols]).astype(np.float32, copy=False)
            
        if len(self.bounded_cols) > 0:
            out[:, self.bounded_cols] = self._minmax_scaler.inverse_transform(out[:, self.bounded_cols]).astype(np.float32, copy=False)
            
        return out


def _scale_3d(X, scaler):
    """
    X: (N, T, F)
    scaler: StandardScaler 또는 CustomMixedScaler (둘 다 .transform(2D) 지원)
    """
    n, t, f = X.shape
    X2 = X.reshape(n * t, f)
    X2s = scaler.transform(X2)
    return X2s.reshape(n, t, f)


def main():
    if not os.path.exists(config.ARTIFACTS_DIR):
        os.makedirs(config.ARTIFACTS_DIR)

    log_folders = [
        f for f in glob.glob(os.path.join(config.DATA_ROOT_DIR, "**", "log_*"), recursive=True)
        if os.path.isdir(f)
    ]
    all_data = []

    print(">>> [PROCESS] Processing Training Logs...")
    for folder in log_folders:
        print(f"Loading: {os.path.basename(folder)}")
        all_data.extend(utils.load_and_process_log(folder))

    if not all_data:
        print("[ERROR] No data found.")
        return

    X_past = np.array([d["x_past"] for d in all_data], dtype=np.float32)              # (N,T,Fpast)
    X_sp = np.array([d["x_sp"] for d in all_data], dtype=np.float32)                  # (N,Tsp,Fsp)
    X_batt_state = np.array([d["x_batt_state"] for d in all_data], dtype=np.float32)  # (N,Fstate)
    y_rft = np.array([d["y_rft"] for d in all_data], dtype=np.float32)
    y_i_mean = np.array([d["y_i_mean"] for d in all_data], dtype=np.float32)
    meta = np.array([d["meta"] for d in all_data], dtype=object)

    # 로그 단위 분할
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(X_past, y_rft, groups=meta[:, 0]))

    # ----------------------------
    # 핵심: 스케일링 제외(또는 범위 지정) 컬럼 정의
    # ----------------------------
    
    
    THRUST_COL = 9
    MOTOR_COLS = [10, 11, 12, 13]

    # ----------------------------
    # Scalers
    # ----------------------------

    # past scaler: thrust_sp는 MinMaxScaler(-2~2) 적용, 나머지는 표준화
    scaler_past = CustomMixedScaler(
        bounded_cols=[THRUST_COL] + MOTOR_COLS,
        feature_range=(-2.0, 2.0)
    ).fit(
        X_past[train_idx].reshape(-1, X_past.shape[2])
    )
    print(">>> [INFO] (thrust_sp,motor_cols MinMax(-2,2) 적용)")


    # sp scaler: 기존대로 전체 표준화 (sp는 [-] 범위가 넓고 단위도 섞일 수 있음)
    scaler_sp = StandardScaler().fit(X_sp[train_idx].reshape(-1, X_sp.shape[2]))


    # batt_state scaler: remaining 제거됨 → 전체 StandardScaler
    scaler_state = StandardScaler().fit(X_batt_state[train_idx])
    print(f">>> [INFO] scaler_batt_state 새로 학습 (remaining 제거, 전체 StandardScaler 적용)")


    # Targets: 회귀 출력은 기존대로 표준화
    scaler_rft = StandardScaler().fit(y_rft[train_idx].reshape(-1, 1))
    scaler_i_mean = StandardScaler().fit(y_i_mean[train_idx].reshape(-1, 1))


    def save_split(idx, name):
        art = config.ARTIFACTS_DIR
    
        np.save(f"{art}/X_past_{name}.npy",
                _scale_3d(X_past[idx], scaler_past))
    
        np.save(f"{art}/X_sp_{name}.npy",
                _scale_3d(X_sp[idx], scaler_sp))
    
        np.save(f"{art}/X_batt_state_{name}.npy",
                scaler_state.transform(X_batt_state[idx]).astype(np.float32))
    
        # 🔥 타겟들은 반드시 2D로 reshape
        np.save(f"{art}/y_rft_{name}.npy",
                scaler_rft.transform(y_rft[idx].reshape(-1, 1)).astype(np.float32))

        np.save(f"{art}/y_i_mean_{name}.npy",
                scaler_i_mean.transform(y_i_mean[idx].reshape(-1, 1)).astype(np.float32))
    
        np.save(f"{art}/meta_{name}.npy", meta[idx])
    
    save_split(train_idx, "train")
    save_split(val_idx, "val")

    # 스케일러 저장 (기존 파일명 유지: 다른 스크립트 호환)
    joblib.dump(scaler_past, f"{config.ARTIFACTS_DIR}/scaler_past.pkl")
    joblib.dump(scaler_sp, f"{config.ARTIFACTS_DIR}/scaler_sp.pkl")
    joblib.dump(scaler_state, f"{config.ARTIFACTS_DIR}/scaler_batt_state.pkl")
    joblib.dump(scaler_rft, f"{config.ARTIFACTS_DIR}/scaler_rft.pkl")
    joblib.dump(scaler_i_mean, f"{config.ARTIFACTS_DIR}/scaler_i_mean.pkl")

    print(f"[SUCCESS] Dataset Ready. Total Samples: {len(X_past)}")


if __name__ == "__main__":
    main()