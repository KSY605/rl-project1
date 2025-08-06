import os
import sys
import argparse
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO

# custom_walker2d_v7_1.py가 있는 경로를 추가합니다.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from custom_walker2d_v7_1 import CustomEnvWrapper

def main():
    """
    학습된 에이전트의 동작을 렌더링하고 비디오로 녹화합니다.
    """
    # --- 1. 커맨드 라인 인자 파싱 ---
    parser = argparse.ArgumentParser(description="Render and record a trained Walker2D agent.")
    parser.add_argument("--model", type=str, required=True, help="렌더링할 학습된 모델(.zip)의 경로")
    parser.add_argument("--curriculum-level", type=int, default=0, help="커리큘럼 레벨 (1-4). 해당 레벨의 환경을 로드합니다.")
    parser.add_argument("--bump-challenge", action='store_true', help="최종 챌린지 환경을 로드합니다.")
    parser.add_argument("--num-bumps", type=int, default=0, help="챌린지 모드에서 활성화할 장애물 개수 (0이면 전부)")
    parser.add_argument("--record", action='store_true', help="에피소드를 비디오 파일(.mp4)로 녹화합니다.")
    args = parser.parse_args()

    # --- 2. 환경 설정 ---
    # 학습 시 사용했던 것과 동일한 인자를 사용하여 환경을 생성합니다.
    # 렌더링을 위해 render_mode='human'으로 설정합니다.
    env = CustomEnvWrapper(
        render_mode='human',
        bump_challenge=args.bump_challenge,
        curriculum_level=args.curriculum_level,
        num_bumps=args.num_bumps
    )

    # --- 3. 비디오 녹화 설정 ---
    if args.record:
        # 모델 파일 이름을 기반으로 비디오 저장 폴더를 생성합니다.
        model_name = os.path.splitext(os.path.basename(args.model))[0]
        video_folder = f"./videos/{model_name}/"
        os.makedirs(video_folder, exist_ok=True)
        
        # Gymnasium의 RecordVideo 래퍼를 환경에 씌웁니다.
        env = RecordVideo(
            env,
            video_folder=video_folder,
            name_prefix=f"render-{model_name}",
            episode_trigger=lambda x: True  # 모든 에피소드를 녹화합니다.
        )
        print(f"녹화가 활성화되었습니다. 비디오는 다음 경로에 저장됩니다: {video_folder}")

    # --- 4. 모델 로드 ---
    try:
        model = PPO.load(args.model, env=env)
        print(f"모델을 성공적으로 로드했습니다: {args.model}")
    except Exception as e:
        print(f"모델 로드 중 오류가 발생했습니다: {e}")
        env.close()
        return

    # --- 5. 렌더링 루프 ---
    # Stable Baselines3는 내부적으로 VecEnv를 사용하므로, 이를 통해 상호작용합니다.
    vec_env = model.get_env()
    obs = vec_env.reset()
    
    total_rewards = 0
    
    print("\n렌더링을 시작합니다. 종료하려면 Ctrl+C를 누르세요.")
    try:
        # 무한 루프를 돌며 에피소드를 계속 실행합니다.
        while True:
            # deterministic=True로 설정하여 가장 확률이 높은 행동을 선택하게 합니다.
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = vec_env.step(action)
            
            total_rewards += rewards[0]

            # 에피소드가 끝나면(dones[0]이 True이면) 보상과 통과한 장애물 수를 출력합니다.
            if dones[0]:
                cleared_bumps = infos[0].get('cleared_bumps', 0)
                print(f"에피소드 종료. 총 보상: {total_rewards:.2f}, 통과한 장애물 수: {cleared_bumps}")
                total_rewards = 0
                # VecEnv는 에피소드가 끝나면 자동으로 다음 에피소드를 위해 리셋합니다.

    except KeyboardInterrupt:
        print("\n사용자에 의해 렌더링이 중단되었습니다.")
    finally:
        # --- 6. 종료 처리 ---
        vec_env.close()
        print("환경이 종료되었습니다.")

if __name__ == "__main__":
    main()