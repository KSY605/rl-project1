import os
import sys
import argparse
import torch as th
import zipfile
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import get_device

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# 커리큘럼 로직이 추가된 환경 파일을 임포트합니다.
from custom_walker2d_v7_1 import CustomEnvWrapper 

# ==================== CURRICULUM: 변경점 1 ====================
# make_env 함수가 curriculum_level을 인자로 받도록 수정합니다.
def make_env(bump_challenge=True, rank=0, seed=0, curriculum_level=0):
# =============================================================
    """
    학습을 위한 환경 생성 함수
    """
    def _init():
        # CustomEnvWrapper를 생성할 때 curriculum_level을 전달합니다.
        env = CustomEnvWrapper(
            render_mode=None, 
            bump_challenge=bump_challenge, 
            curriculum_level=curriculum_level
        )
        env.reset(seed=seed + rank)
        return env
    return _init

def transfer_weights(pretrained_params, new_model_params):
    """
    Pre-trained 모델의 가중치를 새로운 모델로 이식합니다.
    """
    for (new_name, new_param), (old_name, old_param) in zip(new_model_params.items(), pretrained_params.items()):
        if new_param.shape == old_param.shape:
            new_param.data.copy_(old_param.data)
        else:
            if '.0.' in new_name and len(new_param.shape) > 1 and len(old_param.shape) > 1:
                old_obs_dim = old_param.shape[1]
                new_param.data[:, :old_obs_dim] = old_param.data
                print(f"Partially copied weights for input layer: {new_name}")
            else:
                print(f"Skipped mismatched layer: {new_name}. New shape: {new_param.shape}, Old shape: {old_param.shape}")
    return new_model_params


if __name__ == "__main__":
    # --- 설정 (필요시 수정) ---
    N_ENVS = 8 # 병렬 환경 수, CPU 코어 수에 맞춰 조절
    LOG_INTERVAL = 1
    
    parser = argparse.ArgumentParser(description="Fine-tuning Walker2D for Bumps with Curriculum Learning")
    parser.add_argument("--pretrained-model", type=str, required=True, help="사전 학습된 모델의 경로. (예: ./models/walker_flat.zip)")
    
    # ==================== CURRICULUM: 변경점 2 ====================
    # curriculum-level을 커맨드 라인 인자로 추가합니다.
    parser.add_argument(
        "--curriculum-level", 
        type=int, 
        default=0, 
        help="커리큘럼 단계 (1: 5 bumps, 2: 10 bumps, 3: 15 bumps, 0: all bumps)"
    )
    # =============================================================
    
    args = parser.parse_args()

    # 커리큘럼 레벨에 따라 로그 및 저장 폴더 이름을 다르게 설정합니다.
    level_str = f"level{args.curriculum_level}" if args.curriculum_level > 0 else "all_bumps"
    folder_name = f"walker_curriculum_{level_str}"
    log_dir = f"./logs/{folder_name}/"
    save_path = f'./checkpoints/{folder_name}/'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)

    # ==================== CURRICULUM: 변경점 3 ====================
    # SubprocVecEnv를 생성할 때, make_env에 curriculum_level을 전달합니다.
    env = SubprocVecEnv([make_env(rank=i, curriculum_level=args.curriculum_level) for i in range(N_ENVS)])
    # =============================================================
    
    env = VecMonitor(env, filename=os.path.join(log_dir, "monitor.csv"))
    # env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    print("Manually setting network architecture to match the pre-trained model.")
    policy_kwargs = {
        "net_arch": { "pi": [128, 64, 64], "vf": [128, 64, 64] },
        "log_std_init": -1.0
    }
    print(f"Using hardcoded architecture: {policy_kwargs}")

    checkpoint_callback = CheckpointCallback(
        save_freq=max(25000 // N_ENVS, 1),
        save_path=save_path,
        name_prefix="walker_curriculum_model"
    )

    print("Creating a new model with the specified architecture...")
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log=log_dir, 
        policy_kwargs=policy_kwargs,
        device="auto",
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.005,
    )
    
    print(f"\nLoading weights from pre-trained model: {args.pretrained_model}\n")
    
    try:
        with zipfile.ZipFile(args.pretrained_model, "r") as archive:
            policy_file_path = next(name for name in archive.namelist() if name.endswith("policy.pth"))
            with archive.open(policy_file_path, "r") as policy_file:
                pretrained_model_params = th.load(policy_file, map_location=get_device("auto"))
    except Exception as e:
        print(f"Error loading weights from zip file: {e}")
        sys.exit(1)
    
    new_model_policy_params = model.policy.state_dict()
    transfer_weights(pretrained_model_params, new_model_policy_params)
    model.policy.load_state_dict(new_model_policy_params)
    
    print("\nWeight transfer complete. Starting fine-tuning...\n")

    try:
        model.learn(
            total_timesteps=10_000_000, # 각 단계별 학습량은 조절이 필요할 수 있습니다.
            callback=checkpoint_callback,
            log_interval=LOG_INTERVAL,
            reset_num_timesteps=False
        )
    except KeyboardInterrupt:
        print("학습이 중단되었습니다. 현재 모델을 저장합니다.")
    finally:
        model.save(os.path.join(save_path, "final_model"))
        env.close()
