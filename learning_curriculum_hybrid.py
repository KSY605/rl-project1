import os
import sys
import argparse
import torch as th
import zipfile
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import get_device

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# 단계별 보상 설계가 적용된 환경 파일을 임포트합니다.
from custom_walker2d_v7_1_curriculum_hybrid import CustomEnvWrapper 

def make_env(rank=0, seed=0, curriculum_level=0, bump_challenge=False, num_bumps=0):
    """
    학습을 위한 환경 생성 함수
    """
    def _init():
        env = CustomEnvWrapper(
            render_mode=None, 
            bump_challenge=bump_challenge,
            curriculum_level=curriculum_level,
            num_bumps=num_bumps
        )
        env.reset(seed=seed + rank)
        return env
    return _init

def transfer_weights(pretrained_params, new_model_params):
    """
    사전 학습된 모델의 가중치를 새로운 모델로 이식합니다.
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
    N_ENVS = 8
    LOG_INTERVAL = 1
    
    parser = argparse.ArgumentParser(description="Hybrid Curriculum Learning for Walker2D")
    
    # ==================== CURRICULUM: 변경점 1 ====================
    # pretrained-model 인자를 선택적으로 변경 (required=False, default=None)
    parser.add_argument("--pretrained-model", type=str, default=None, help="사전 학습된 모델의 경로 (선택 사항)")
    # =============================================================
    
    parser.add_argument("--curriculum-level", type=int, default=0, help="난이도별 커리큘럼 단계 (1-4). 0이면 비활성화.")
    parser.add_argument("--bump-challenge", action='store_true', help="최종 챌린지 환경을 로드할지 여부.")
    parser.add_argument("--num-bumps", type=int, default=0, help="챌린지 환경에서 활성화할 장애물 개수 (0이면 전부).")
    
    args = parser.parse_args()

    # 로그 및 저장 폴더 이름 설정
    if args.curriculum_level > 0:
        folder_name = f"walker_curriculum_level{args.curriculum_level}"
    elif args.bump_challenge:
        num_str = f"{args.num_bumps}bumps" if args.num_bumps > 0 else "all_bumps"
        folder_name = f"walker_challenge_{num_str}"
    else:
        folder_name = "walker_default"
        
    log_dir = f"./logs/{folder_name}/"
    save_path = f'./checkpoints/{folder_name}/'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)

    # 환경 생성
    env = SubprocVecEnv([make_env(
        rank=i, 
        curriculum_level=args.curriculum_level,
        bump_challenge=args.bump_challenge,
        num_bumps=args.num_bumps
    ) for i in range(N_ENVS)])
    
    env = VecMonitor(env, filename=os.path.join(log_dir, "monitor.csv"))

    # 고정된 정책 네트워크 구조
    policy_kwargs = {
        "net_arch": { "pi": [128, 64, 64], "vf": [128, 64, 64] },
        "log_std_init": -1.0
    }

    checkpoint_callback = CheckpointCallback(
        save_freq=max(25000 // N_ENVS, 1),
        save_path=save_path,
        name_prefix="walker_model"
    )

    # ==================== CURRICULUM: 변경점 2 ====================
    # 모델 생성 및 가중치 로드 로직 변경
    
    # 먼저, 빈 PPO 모델을 생성합니다.
    model = PPO(
        "MlpPolicy", env, verbose=1, tensorboard_log=log_dir, 
        policy_kwargs=policy_kwargs, device="auto", learning_rate=1e-4,
        n_steps=2048, batch_size=64, gamma=0.99, gae_lambda=0.95, ent_coef=0.005,
    )

    # --pretrained-model 인자가 주어졌을 경우에만 가중치를 로드합니다.
    if args.pretrained_model:
        print(f"\nLoading weights from pre-trained model: {args.pretrained_model}\n")
        try:
            with zipfile.ZipFile(args.pretrained_model, "r") as archive:
                policy_file_path = next(name for name in archive.namelist() if name.endswith("policy.pth"))
                with archive.open(policy_file_path, "r") as policy_file:
                    pretrained_model_params = th.load(policy_file, map_location=get_device("auto"))
        except Exception as e:
            print(f"Error loading weights from zip file: {e}"); sys.exit(1)
        
        new_model_policy_params = model.policy.state_dict()
        transfer_weights(pretrained_model_params, new_model_policy_params)
        model.policy.load_state_dict(new_model_policy_params)
        
        print("\nWeight transfer complete. Starting fine-tuning...\n")
    else:
        # 인자가 없으면, 처음부터 학습을 시작합니다.
        print("\nNo pre-trained model provided. Starting training from scratch.\n")
    # =============================================================

    try:
        model.learn(
            total_timesteps=10_000_000, callback=checkpoint_callback,
            log_interval=LOG_INTERVAL, reset_num_timesteps=False
        )
    except KeyboardInterrupt:
        print("학습이 중단되었습니다. 현재 모델을 저장합니다.")
    finally:
        model.save(os.path.join(save_path, "final_model"))
        env.close()
