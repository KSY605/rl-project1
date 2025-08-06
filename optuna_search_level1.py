import os
import sys
import numpy as np
import gymnasium as gym
import torch as th
import optuna

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from typing import List, Tuple, Dict, Any

# ==============================================================================
# 1. 제공된 파일의 핵심 클래스 및 함수 통합
# custom_walker2d_v7_1_curriculum_hybrid.py의 CustomEnvWrapper를 가져옵니다.
# Optuna Trial에서 제안된 하이퍼파라미터를 사용하도록 수정합니다.
# ==============================================================================

class CustomEnvWrapperForOptuna(gym.Wrapper):
    """
    Optuna 최적화를 위해 수정된 Walker2d 환경 래퍼.
    trial 객체로부터 하이퍼파라미터를 받아 보상 함수를 동적으로 구성합니다.
    """
    def __init__(self, render_mode=None, curriculum_level: int = 0, trial: optuna.Trial = None):
        # --- 환경 로드 ---
        # asset 폴더와 xml 파일이 없어도 오류가 나지 않도록 기본 환경을 사용합니다.
        try:
            repo_root = os.path.dirname(os.path.abspath(__file__))
            asset_dir = os.path.join(repo_root, "asset")
            xml_file = os.path.join(asset_dir, f"curriculum_level{curriculum_level}.xml")
            if not os.path.exists(xml_file):
                # print(f"--- WARNING: {xml_file} not found. Using default Walker2d-v5. ---")
                xml_file = None
        except Exception:
            xml_file = None

        env = gym.make(
            "Walker2d-v5", xml_file=xml_file, render_mode=render_mode,
            exclude_current_positions_from_observation=False, frame_skip=10, healthy_z_range=(0.4, 10.0),
        )
        super().__init__(env)

        self.base_env = env.unwrapped
        base_model = self.base_env.model
        
        # --- 장애물 정보 초기화 (원본 코드와 동일) ---
        all_bump_geoms = sorted(
            [(i, base_model.geom(i).name) for i in range(base_model.ngeom)
             if base_model.geom(i).name and base_model.geom(i).name.startswith("bump")],
            key=lambda item: int(item[1].replace("bump", "")))
        self.active_bump_geom_ids = [item[0] for item in all_bump_geoms]
        self.foot_geom_ids = {base_model.geom(name).id for name in ["foot_geom", "foot_left_geom"]}
        
        # --- 상태 변수 초기화 ---
        self.cleared_bumps_count = 0
        self.current_bump_target_x = np.inf
        
        # --- Optuna 하이퍼파라미터 설정 ---
        self.trial = trial
        if self.trial:
            self.W_FORWARD = self.trial.suggest_float("W_FORWARD", 0.5, 4.0)
            self.W_ALIVE = self.trial.suggest_float("W_ALIVE", 0.0, 0.5)
            self.W_CTRL = self.trial.suggest_float("W_CTRL", -0.05, -0.001, log=True)
            self.W_STABILITY_PENALTY = self.trial.suggest_float("W_STABILITY_PENALTY", -0.1, -0.001, log=True)
            self.SUCCESS_BONUS = self.trial.suggest_float("SUCCESS_BONUS", 10.0, 100.0)
        else: # 기본값 설정
            self.W_FORWARD, self.W_ALIVE, self.W_CTRL, self.W_STABILITY_PENALTY, self.SUCCESS_BONUS = 2.0, 0.1, -0.01, -0.02, 50.0

        sample_obs, _ = self.reset()
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=sample_obs.shape, dtype=np.float64)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.cleared_bumps_count = 0
        
        next_bump_info = self._get_next_n_bumps_info(n=1)
        if next_bump_info and next_bump_info[0][3] != -1.0:
            self.current_bump_target_x = next_bump_info[0][3]
        else:
            self.current_bump_target_x = np.inf
        return self.custom_observation(obs), info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        custom_obs = self.custom_observation(obs)
        custom_reward, success_bonus = self._calculate_rewards(obs, action)
        info['cleared_bumps'] = self.cleared_bumps_count
        if success_bonus > 0: info['event'] = 'BUMP_CLEARED'
        return custom_obs, custom_reward, terminated, truncated, info

    def _calculate_rewards(self, obs: np.ndarray, action: np.ndarray) -> Tuple[float, float]:
        """
        Optuna trial에서 제안된 가중치를 사용하여 보상 함수를 계산합니다.
        """
        z_torso, vel_x, angvel_torso = obs[1], obs[9], obs[11]

        # --- 1. 기반 보상 ---
        W_FALL_PENALTY = -50.0
        reward_forward = self.W_FORWARD * vel_x
        reward_alive = self.W_ALIVE
        penalty_control = self.W_CTRL * np.sum(np.square(action))
        penalty_stability = self.W_STABILITY_PENALTY * np.square(angvel_torso)
        penalty_fall = W_FALL_PENALTY if z_torso < 0.7 else 0.0
        base_reward = reward_forward + reward_alive + penalty_control + penalty_stability + penalty_fall

        # --- 2. 장애물 통과 성공 보너스 ---
        MIN_SUCCESS_HEIGHT = 1.0
        success_bonus = 0.0
        next_bump_info = self._get_next_n_bumps_info(n=1)
        next_bump_x_pos = next_bump_info[0][3] if next_bump_info and len(next_bump_info) > 0 else -1.0
        
        if self.current_bump_target_x > 0 and self.current_bump_target_x != next_bump_x_pos and z_torso > MIN_SUCCESS_HEIGHT:
            success_bonus = self.SUCCESS_BONUS
            self.cleared_bumps_count += 1
        self.current_bump_target_x = next_bump_x_pos

        # --- 최종 보상 합산 ---
        total_reward = base_reward + success_bonus
        return total_reward, success_bonus

    def _get_next_n_bumps_info(self, n: int = 4) -> List[Tuple[float, float, float, float]]:
        # 이 함수는 장애물 정보를 관측에 추가하기 위해 필요합니다.
        # XML 파일이 없으면 빈 리스트를 반환합니다.
        if not self.active_bump_geom_ids:
            return [[-1.0, 0.0, 0.0, -1.0]] * n
            
        data, model = self.base_env.data, self.base_env.model
        walker_x = data.qpos[0]
        upcoming_bumps = []
        for gid in self.active_bump_geom_ids:
            current_bump_x_pos = data.geom_xpos[gid][0]
            dx = current_bump_x_pos - walker_x
            if dx >= 0.0:
                width = model.geom_size[gid][0] 
                height = model.geom_size[gid][2]
                upcoming_bumps.append([dx, height, width, current_bump_x_pos])
        upcoming_bumps.sort(key=lambda bump: bump[0])
        result = upcoming_bumps[:n]
        padding_info = [-1.0, 0.0, 0.0, -1.0]
        while len(result) < n:
            result.append(padding_info)
        return result

    def custom_observation(self, obs: np.ndarray) -> np.ndarray:
        next_bumps_info = self._get_next_n_bumps_info(n=4)
        obs_bumps_info = [info[:3] for info in next_bumps_info]
        flat_bumps_info = np.array(obs_bumps_info, dtype=np.float64).flatten()
        return np.concatenate([obs, flat_bumps_info])

# ==============================================================================
# 2. Optuna Objective 함수 정의
# ==============================================================================

def make_env(rank=0, seed=0, curriculum_level=1, trial=None):
    """학습을 위한 환경 생성 함수"""
    def _init():
        # asset 폴더 생성 (XML 파일이 없어도 경로 문제 방지)
        os.makedirs("./asset", exist_ok=True)
        env = CustomEnvWrapperForOptuna(curriculum_level=curriculum_level, trial=trial)
        env.reset(seed=seed + rank)
        return env
    return _init

def objective(trial: optuna.Trial) -> float:
    """
    Optuna가 최적화할 목적 함수.
    주어진 trial에 대해 모델을 학습하고 평가 점수를 반환합니다.
    """
    N_ENVS = 8  # 최적화 속도를 위해 병렬 환경 수를 줄입니다.
    TRAIN_TIMESTEPS = 50_000 # 각 trial 당 학습 스텝
    EVAL_TIMESTEPS = 5_000  # 평가 스텝

    # --- PPO 하이퍼파라미터 제안 ---
    ppo_params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "n_steps": trial.suggest_categorical("n_steps", [512, 1024, 2048]),
        "gamma": trial.suggest_float("gamma", 0.9, 0.9999),
        "ent_coef": trial.suggest_float("ent_coef", 1e-8, 1e-1, log=True),
        "batch_size": 64,
    }
    
    # --- 환경 생성 ---
    # SubprocVecEnv를 사용하여 병렬 학습을 수행합니다.
    env = SubprocVecEnv([make_env(rank=i, curriculum_level=1, trial=trial) for i in range(N_ENVS)])
    env = VecMonitor(env)

    # --- 모델 생성 및 학습 ---
    model = PPO("MlpPolicy", env, verbose=0, **ppo_params)
    
    try:
        model.learn(total_timesteps=TRAIN_TIMESTEPS)
        
        # --- 모델 평가 ---
        eval_env = make_env(rank=N_ENVS, curriculum_level=1, trial=trial)() # 평가용 단일 환경
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
        
        # 사용한 자원 정리
        env.close()
        eval_env.close()

    except Exception as e:
        print(f"An error occurred during training/evaluation: {e}")
        # 오류 발생 시, 낮은 점수를 반환하여 해당 trial을 실패로 처리
        return -1e9

    # Optuna는 반환된 값을 최대화하는 방향으로 탐색합니다.
    return mean_reward

# ==============================================================================
# 3. Optuna Study 실행
# ==============================================================================

if __name__ == "__main__":
    # Optuna가 탐색 기록을 저장할 데이터베이스 파일 설정
    study_name = "ppo-walker-level1-study"
    storage_name = f"sqlite:///{study_name}.db"
    
    # TPESampler: 이전 결과를 바탕으로 유망한 하이퍼파라미터 공간을 효율적으로 탐색
    sampler = optuna.samplers.TPESampler()
    
    # 새로운 study를 생성하거나, 기존 study가 있으면 이어서 탐색합니다.
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        sampler=sampler,
        direction="maximize", # objective 함수가 반환하는 점수(mean_reward)를 최대화
        load_if_exists=True,
    )

    print(f"Using storage: {storage_name}")
    print(f"Number of finished trials: {len(study.trials)}")

    try:
        # n_trials 만큼 최적화를 실행합니다.
        study.optimize(objective, n_trials=50, n_jobs=-1) # n_jobs=-1 로 설정하면 가능한 모든 CPU 코어 사용
    except KeyboardInterrupt:
        print("Optimization stopped by user.")

    # --- 결과 출력 ---
    print("\n==================================================")
    print("Optimization Finished!")
    print(f"Number of finished trials: {len(study.trials)}")
    
    best_trial = study.best_trial
    print(f"Best trial value (mean reward): {best_trial.value:.2f}")
    
    print("\nBest hyperparameters found:")
    for key, value in best_trial.params.items():
        print(f"  - {key}: {value}")
    
    # 시각화 결과를 확인하고 싶다면 아래 주석을 해제하세요.
    # pip install plotly kaleido
    # if optuna.visualization.is_available():
    #     fig = optuna.visualization.plot_optimization_history(study)
    #     fig.write_image("optuna_history.png")
    #     fig = optuna.visualization.plot_param_importances(study)
    #     fig.write_image("optuna_importances.png")

