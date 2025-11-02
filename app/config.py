import os

MODEL_PATH = os.getenv("MODEL_PATH", "./models/e5_stage_pairs_triplets")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1.0.0")
DEVICE = os.getenv("DEVICE", "cpu") 

# 재랭킹 가중치
W_CAT = float(os.getenv("W_CAT", "0.15"))
W_TIME = float(os.getenv("W_TIME", "0.05"))
W_GOAL = float(os.getenv("W_GOAL", "0.05"))

TOP_K_DEFAULT = int(os.getenv("TOP_K_DEFAULT", "5"))
CAND_N = int(os.getenv("CAND_N", "50"))  # 1차 상위 50개 후보