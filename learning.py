import os
import sys
import argparse
import torch as th
import zipfile
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import get_device

# custom_walker2d.py가 있는 디렉토리를 파이썬 경로에 추가합니다.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# 폭(width) 정보까지 포함된 최신 버전의 환경을 사용합니다.
from custom_walker2d_v4 import CustomEnvWrapper 

def make_env(bump_challenge=True, rank=0, seed=0):
    """
    학습을 위한 환경 생성 함수
    """
    def _init():
        # Fine-tuning은 항상 bump가 있는 환경에서 진행합니다.
        env = CustomEnvWrapper(render_mode=None, bump_challenge=bump_challenge)
        env.reset(seed=seed + rank)
        return env
    return _init

def transfer_weights(pretrained_params, new_model_params):
    """
    Pre-trained 모델의 가중치를 새로운 모델로 이식합니다.
    Observation Space의 크기가 다른 입력층을 처리하는 것이 핵심입니다.
    """
    for (new_name, new_param), (old_name, old_param) in zip(new_model_params.items(), pretrained_params.items()):
        # 레이어 이름과 모양이 같다면 그대로 복사
        if new_param.shape == old_param.shape:
            new_param.data.copy_(old_param.data)
        else:
            # 입력층(Observation을 직접 받는 첫 번째 레이어) 처리
            if '.0.' in new_name and len(new_param.shape) > 1 and len(old_param.shape) > 1:
                # pre-trained 모델의 observation 차원 수를 가져옵니다.
                old_obs_dim = old_param.shape[1]
                # 새로운 모델의 가중치 텐서에서, 기존 observation에 해당하는 부분만 복사합니다.
                new_param.data[:, :old_obs_dim] = old_param.data
                print(f"Partially copied weights for input layer: {new_name}")
            else:
                # 입력층이 아닌데도 모양이 다르면 (net_arch가 다르면) 경고를 출력합니다.
                print(f"Skipped mismatched layer: {new_name}. New shape: {new_param.shape}, Old shape: {old_param.shape}")
    return new_model_params


if __name__ == "__main__":
    # --- 설정 (필요시 수정) ---
    N_ENVS = 8
    LOG_INTERVAL = 1
    
    parser = argparse.ArgumentParser(description="Fine-tuning Walker2D for Bumps")
    parser.add_argument("--pretrained-model", type=str, required=True, help="잘 걷는 pre-trained 모델의 경로. (예: ./models/walker_flat.zip)")
    args = parser.parse_args()

    folder_name = "walker_finetuned_v3"
    log_dir = f"./logs/{folder_name}/"
    save_path = f'./checkpoints/{folder_name}/'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)

    env = SubprocVecEnv([make_env(rank=i) for i in range(N_ENVS)])
    env = VecMonitor(env, filename=os.path.join(log_dir, "monitor.csv"))

    # 정규화 적용 (obs, reward 정규화)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # --- 최종 오류 해결 ---
    # 불안정한 자동 아키텍처 추출 로직을 완전히 제거하고,
    # 오류 로그를 통해 확인된 Pre-trained 모델의 신경망 구조를 직접 명시합니다.
    # 이 방법이 버전 차이에 상관없이 가장 확실하게 동작합니다.
    print("Manually setting network architecture to match the pre-trained model.")
    policy_kwargs = {
        "net_arch": {
            "pi": [128, 64, 64], # Policy Network 구조
            "vf": [128, 64, 64]  # Value Network 구조
        },
        "log_std_init": -1.0
    }
    print(f"Using hardcoded architecture: {policy_kwargs}")
    # ---------------------

    checkpoint_callback = CheckpointCallback(
        save_freq=max(25000 // N_ENVS, 1),
        save_path=save_path,
        name_prefix="walker_finetuned_model"
    )

    # 2. 지정된 신경망 구조를 사용하여 새로운 모델을 생성합니다.
    print("Creating a new model with the specified architecture...")
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log=log_dir, 
        policy_kwargs=policy_kwargs, # 직접 지정한 policy_kwargs를 사용
        device="auto",
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
    )
    
    print(f"\nLoading weights from pre-trained model: {args.pretrained_model}\n")
    
    # 3. Pre-trained 모델에서 가중치(weights)만 불러옵니다.
    try:
        with zipfile.ZipFile(args.pretrained_model, "r") as archive:
            policy_file_path = next(name for name in archive.namelist() if name.endswith("policy.pth"))
            with archive.open(policy_file_path, "r") as policy_file:
                pretrained_model_params = th.load(policy_file, map_location=get_device("auto"))
    except Exception as e:
        print(f"Error loading weights from zip file: {e}")
        sys.exit(1)
    
    # 4. 가중치를 새로운 모델에 이식합니다.
    new_model_policy_params = model.policy.state_dict()
    transfer_weights(pretrained_model_params, new_model_policy_params)
    model.policy.load_state_dict(new_model_policy_params)
    
    print("\nWeight transfer complete. Starting fine-tuning...\n")

    try:
        model.learn(
            total_timesteps=100_000_000,
            callback=checkpoint_callback,
            log_interval=LOG_INTERVAL,
            reset_num_timesteps=False
        )
    except KeyboardInterrupt:
        print("학습이 중단되었습니다. 현재 모델을 저장합니다.")
    finally:
        model.save(os.path.join(save_path, "final_finetuned_model"))
        env.close()


# import os
# import sys
# import argparse
# import torch as th
# import zipfile
# import json
# import ast # To safely evaluate string representations of dictionaries
# from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
# from stable_baselines3.common.callbacks import CheckpointCallback
# from stable_baselines3.common.utils import get_device

# # custom_walker2d.py가 있는 디렉토리를 파이썬 경로에 추가합니다.
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# # 폭(width) 정보까지 포함된 최신 버전의 환경을 사용합니다.
# from custom_walker2d_v2 import CustomEnvWrapper 

# def make_env(bump_challenge=True, rank=0, seed=0):
#     """
#     학습을 위한 환경 생성 함수
#     """
#     def _init():
#         # Fine-tuning은 항상 bump가 있는 환경에서 진행합니다.
#         env = CustomEnvWrapper(render_mode=None, bump_challenge=bump_challenge)
#         env.reset(seed=seed + rank)
#         return env
#     return _init

# def transfer_weights(pretrained_params, new_model_params):
#     """
#     Pre-trained 모델의 가중치를 새로운 모델로 이식합니다.
#     Observation Space의 크기가 다른 입력층을 처리하는 것이 핵심입니다.
#     """
#     for (new_name, new_param), (old_name, old_param) in zip(new_model_params.items(), pretrained_params.items()):
#         # 레이어 이름과 모양이 같다면 그대로 복사
#         if new_param.shape == old_param.shape:
#             new_param.data.copy_(old_param.data)
#         else:
#             # 입력층(Observation을 직접 받는 첫 번째 레이어) 처리
#             if '.0.' in new_name and len(new_param.shape) > 1 and len(old_param.shape) > 1:
#                 # pre-trained 모델의 observation 차원 수를 가져옵니다.
#                 old_obs_dim = old_param.shape[1]
#                 # 새로운 모델의 가중치 텐서에서, 기존 observation에 해당하는 부분만 복사합니다.
#                 new_param.data[:, :old_obs_dim] = old_param.data
#                 print(f"Partially copied weights for input layer: {new_name}")
#             else:
#                 # 입력층이 아닌데도 모양이 다르면 (net_arch가 다르면) 경고를 출력합니다.
#                 print(f"Skipped mismatched layer: {new_name}. New shape: {new_param.shape}, Old shape: {old_param.shape}")
#     return new_model_params


# if __name__ == "__main__":
#     # --- 설정 (필요시 수정) ---
#     N_ENVS = 8
#     LOG_INTERVAL = 1
    
#     parser = argparse.ArgumentParser(description="Fine-tuning Walker2D for Bumps")
#     parser.add_argument("--pretrained-model", type=str, required=True, help="잘 걷는 pre-trained 모델의 경로. (예: ./models/walker_flat.zip)")
#     args = parser.parse_args()

#     folder_name = "walker_finetuned_paqur"
#     log_dir = f"./logs/{folder_name}/"
#     save_path = f'./checkpoints/{folder_name}/'
#     os.makedirs(log_dir, exist_ok=True)
#     os.makedirs(save_path, exist_ok=True)

#     env = SubprocVecEnv([make_env(rank=i) for i in range(N_ENVS)])
#     env = VecMonitor(env, filename=os.path.join(log_dir, "monitor.csv"))

#     # --- 최종 오류 해결 ---
#     # 1. Pre-trained 모델 파일(.zip)에서 신경망 구조 정보를 텍스트로 직접 추출합니다.
#     print(f"Extracting architecture from {args.pretrained_model}...")
#     try:
#         with zipfile.ZipFile(args.pretrained_model, "r") as archive:
#             # zip 파일 내의 'data' 파일을 텍스트로 읽어옵니다.
#             json_data_str = archive.read("data").decode("utf-8")
            
#             # JSON 데이터로 변환합니다.
#             data = json.loads(json_data_str)
            
#             # 오래된 SB3 버전에서는 policy_kwargs가 문자열로 저장되어 있을 수 있습니다.
#             # 이 문자열을 ast.literal_eval을 사용해 안전하게 파이썬 딕셔너리로 변환합니다.
#             policy_kwargs_str = data.get("policy_kwargs", "{}")
#             policy_kwargs = ast.literal_eval(policy_kwargs_str)
            
#             print(f"Successfully extracted architecture: {policy_kwargs.get('net_arch')}")

#     except Exception as e:
#         print(f"Error extracting architecture from zip file: {e}")
#         print("Could not automatically determine the network architecture.")
#         print("This might be due to an incompatible model version.")
#         sys.exit(1)
#     # ---------------------

#     checkpoint_callback = CheckpointCallback(
#         save_freq=max(25000 // N_ENVS, 1),
#         save_path=save_path,
#         name_prefix="walker_finetuned_model"
#     )

#     # 2. 추출한 신경망 구조를 사용하여 새로운 모델을 생성합니다.
#     print("Creating a new model with the extracted architecture...")
#     model = PPO(
#         "MlpPolicy", 
#         env, 
#         verbose=1, 
#         tensorboard_log=log_dir, 
#         policy_kwargs=policy_kwargs, # 추출한 policy_kwargs를 그대로 사용
#         device="auto",
#         learning_rate=1e-4,
#         n_steps=2048,
#         batch_size=64,
#         gamma=0.99,
#         gae_lambda=0.95,
#     )
    
#     print(f"\nLoading weights from pre-trained model: {args.pretrained_model}\n")
    
#     # 3. Pre-trained 모델에서 가중치(weights)만 불러옵니다.
#     try:
#         with zipfile.ZipFile(args.pretrained_model, "r") as archive:
#             policy_file_path = next(name for name in archive.namelist() if name.endswith("policy.pth"))
#             with archive.open(policy_file_path, "r") as policy_file:
#                 pretrained_model_params = th.load(policy_file, map_location=get_device("auto"))
#     except Exception as e:
#         print(f"Error loading weights from zip file: {e}")
#         sys.exit(1)
    
#     # 4. 가중치를 새로운 모델에 이식합니다.
#     new_model_policy_params = model.policy.state_dict()
#     transfer_weights(pretrained_model_params, new_model_policy_params)
#     model.policy.load_state_dict(new_model_policy_params)
    
#     print("\nWeight transfer complete. Starting fine-tuning...\n")

#     try:
#         model.learn(
#             total_timesteps=100_000_000,
#             callback=checkpoint_callback,
#             log_interval=LOG_INTERVAL,
#             reset_num_timesteps=False
#         )
#     except KeyboardInterrupt:
#         print("학습이 중단되었습니다. 현재 모델을 저장합니다.")
#     finally:
#         model.save(os.path.join(save_path, "final_finetuned_model"))
#         env.close()
