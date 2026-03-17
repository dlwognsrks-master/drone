import os

# ==========================================
# 1. 경로 설정
# ==========================================
DATA_ROOT_DIR = '/home/jaehun/drone_real_dataset_22/csv'
TEST_LOG_ROOT = '/home/jaehun/drone_real_dataset_22/test_data'
ARTIFACTS_DIR = './artifacts'
ARTIFACTS_DIR_offline = './artifacts_offline'

# ==========================================
# 2. 데이터 구조 및 전처리 설정
# ==========================================
# Past (5Hz 기준 0.2s)
PAST_SEQ_LEN = 25     # 과거 5초
PAST_STRIDE = 1        # stride : 보폭 / 데이터를 몇 칸씩 건너뛰며 가져올 것인가 하는 간격

# Near-future setpoint stream (k용)
SP_SEQ_LEN = 5         # 근미래 1초 (5Hz * 1s = 5)
SP_FEATURES = 4        # [traj_vx, traj_vy, traj_vz, thrust_sp] 권장

# ΔI 라벨 계산 horizon (5Hz 기준)
DELTA_I_H = 5          

# 기존 future segment(길이 편향 이슈 때문에 RFT에는 쓰지 않음)
FUTURE_SEG_SEQ_LEN = 50
CHUNK_SIZE = 10

# Past features (현재 utils.py 기준 13개)
PAST_FEATURES = 21

# ==========================================
# 3. 딥러닝 모델 하이퍼파라미터
# ==========================================
BATCH_SIZE = 64
EPOCHS = 200
LEARNING_RATE = 0.001
