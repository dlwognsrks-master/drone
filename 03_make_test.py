import os
import glob
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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


def scale_3d(X, scaler):
    """
    (N, T, F) 형태의 시계열 데이터를 scaler에 맞게 변환
    """
    n, t, f = X.shape
    X2 = X.reshape(n * t, f)
    X2s = scaler.transform(X2)
    return X2s.reshape(n, t, f)

def main():
    art = config.ARTIFACTS_DIR
    os.makedirs(art, exist_ok=True)

    log_folders = [
        f for f in glob.glob(os.path.join(config.TEST_LOG_ROOT, "**", "log_*"), recursive=True)
        if os.path.isdir(f)
    ]

    all_data = []
    print(">>> [TEST] Processing Test Logs (No Leakage)...")
    for folder in log_folders:
        all_data.extend(utils.load_and_process_log(folder))

    if not all_data:
        print("[ERROR] No test samples generated. Check TEST_LOG_ROOT or CSV availability.")
        return

    X_past = np.array([d["x_past"] for d in all_data], dtype=np.float32)
    X_sp = np.array([d["x_sp"] for d in all_data], dtype=np.float32)
    X_batt_state = np.array([d["x_batt_state"] for d in all_data], dtype=np.float32)

    y_rft = np.array([d["y_rft"] for d in all_data], dtype=np.float32)
    y_i_mean = np.array([d["y_i_mean"] for d in all_data], dtype=np.float32)

    meta = np.array([d["meta"] for d in all_data], dtype=object)

    try:
        s_past = joblib.load(os.path.join(art, "scaler_past.pkl"))
        s_sp = joblib.load(os.path.join(art, "scaler_sp.pkl"))
        s_batt = joblib.load(os.path.join(art, "scaler_batt_state.pkl"))
        s_rft = joblib.load(os.path.join(art, "scaler_rft.pkl"))
        s_im = joblib.load(os.path.join(art, "scaler_i_mean.pkl"))
    except FileNotFoundError as e:
        print(f"[ERROR] Missing scaler file: {e}")
        print("[HINT] Run 01_data_processor.py first to generate scalers.")
        return

    X_past_s = scale_3d(X_past, s_past)
    X_sp_s = scale_3d(X_sp, s_sp)
    X_batt_s = s_batt.transform(X_batt_state)

    # ✅ targets: 반드시 2D
    y_rft_s = s_rft.transform(y_rft.reshape(-1, 1)).astype(np.float32)
    y_im_s  = s_im.transform(y_i_mean.reshape(-1, 1)).astype(np.float32)

    np.save(os.path.join(art, "X_past_test.npy"), X_past_s)
    np.save(os.path.join(art, "X_sp_test.npy"), X_sp_s)
    np.save(os.path.join(art, "X_batt_state_test.npy"), X_batt_s)

    np.save(os.path.join(art, "y_rft_test.npy"), y_rft_s)
    np.save(os.path.join(art, "y_i_mean_test.npy"), y_im_s)

    np.save(os.path.join(art, "meta_test.npy"), meta)

    print("=" * 60)
    print("[SUCCESS] Test Dataset Ready!")
    print(f"  Samples: {len(X_past)}")
    print(f"  Saved to: {art}")
    print("=" * 60)

if __name__ == "__main__":
    main()