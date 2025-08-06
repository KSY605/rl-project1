import optuna
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

# custom_walker2d_v8.py가 같은 디렉토리에 있어야 합니다.
from custom_walker2d_v7 import CustomEnvWrapper

# 최적화를 위한 상수
N_TRIALS = 50  # 총 시도 횟수
N_TRAIN_TIMESTEPS = 25000  # 각 trial 당 학습 스텝 수
N_EVAL_EPISODES = 10  # 각 trial 당 평가 에피소드 수
N_ENVS = 4 # 병렬로 실행할 환경 수

def objective(trial: optuna.Trial) -> float:
    """
    Optuna의 각 trial에서 실행될 목적 함수입니다.
    하이퍼파라미터를 제안하고, 모델을 학습시킨 후, 성능을 평가하여 반환합니다.
    """
    print(f"\n===== Trial {trial.number} 시작 =====")

    # 1. 최적화할 하이퍼파라미터(보상 가중치)의 범위를 제안합니다.
    reward_weights = {
        "W_FORWARD": trial.suggest_float("W_FORWARD", 0.5, 2.0),
        "W_CROUCH": trial.suggest_float("W_CROUCH", 2.0, 10.0),
        "W_SLOW_DOWN": trial.suggest_float("W_SLOW_DOWN", 2.0, 10.0),
        "W_SYMMETRY": trial.suggest_float("W_SYMMETRY", 2.0, 10.0),
        "W_CLEARANCE": trial.suggest_float("W_CLEARANCE", 10.0, 30.0),
        "W_JUMP": trial.suggest_float("W_JUMP", 1.0, 5.0),
        "W_PARKOUR": trial.suggest_float("W_PARKOUR", 10.0, 25.0),
    }
    print(f"제안된 가중치: {reward_weights}")

    # 2. 제안된 하이퍼파라미터로 커스텀 환경을 생성합니다.
    # make_vec_env를 사용하여 여러 환경을 병렬로 실행하여 학습 속도를 높입니다.
    env = make_vec_env(
        CustomEnvWrapper,
        n_envs=N_ENVS,
        env_kwargs={'bump_challenge': True, 'reward_weights': reward_weights}
    )

    # 3. PPO 모델을 생성하고 학습시킵니다.
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        # PPO의 하이퍼파라미터도 Optuna로 최적화할 수 있습니다.
        # learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        # n_steps=trial.suggest_int("n_steps", 256, 2048, step=128),
    )
    
    print(f"Trial {trial.number}: 학습 시작... (Timesteps: {N_TRAIN_TIMESTEPS})")
    model.learn(total_timesteps=N_TRAIN_TIMESTEPS, progress_bar=True)
    
    # 4. 학습된 모델을 평가합니다.
    # 평가용 환경은 하나만 생성합니다.
    eval_env = CustomEnvWrapper(bump_challenge=True, reward_weights=reward_weights)
    
    # 통과한 장애물 개수를 직접 계산하여 평가 지표로 사용합니다.
    total_cleared_bumps = 0
    for _ in range(N_EVAL_EPISODES):
        obs, _ = eval_env.reset()
        terminated = False
        truncated = False
        while not terminated and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = eval_env.step(action)
        total_cleared_bumps += info.get('cleared_bumps', 0)
    
    mean_cleared_bumps = total_cleared_bumps / N_EVAL_EPISODES
    
    print(f"Trial {trial.number} 평가 완료. 평균 통과 Bump 개수: {mean_cleared_bumps:.2f}")
    
    env.close()
    eval_env.close()

    # 5. 평가 결과를 반환합니다. Optuna는 이 값을 최대화하는 방향으로 탐색합니다.
    return mean_cleared_bumps


if __name__ == "__main__":
    # Optuna study를 생성합니다.
    # maximize: objective 함수가 반환하는 값을 최대화하는 것이 목표
    # pruner: 성능이 좋지 않은 trial을 조기에 중단시켜 최적화 속도를 높임
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    )

    try:
        # 최적화를 실행합니다.
        study.optimize(objective, n_trials=N_TRIALS, timeout=3600) # 1시간 제한
    except KeyboardInterrupt:
        print("최적화가 사용자에 의해 중단되었습니다.")

    # 최적화 결과를 출력합니다.
    print("\n\n===== 최적화 결과 =====")
    print(f"총 Trial 횟수: {len(study.trials)}")
    
    best_trial = study.best_trial
    print(f"최고 점수 (평균 통과 Bump): {best_trial.value:.4f}")
    
    print("최적 하이퍼파라미터:")
    for key, value in best_trial.params.items():
        print(f"  - {key}: {value}")

