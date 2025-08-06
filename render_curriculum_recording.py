import os
import sys
import argparse
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO

# custom_walker2d_v7_1_curriculum_hybrid.py가 있는 경로를 추가합니다.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from custom_walker2d_v7_1_curriculum_hybrid import CustomEnvWrapper

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

    # ==================== 오류 수정 ====================
    # --record 플래그에 따라 render_mode를 동적으로 설정합니다.
    # 녹화 시에는 'rgb_array', 직접 볼 때는 'human'을 사용합니다.
    render_mode = "rgb_array" if args.record else "human"
    # =================================================

    # --- 2. 환경 설정 ---
    env = CustomEnvWrapper(
        render_mode=render_mode, # 수정된 render_mode를 사용
        bump_challenge=args.bump_challenge,
        curriculum_level=args.curriculum_level,
        num_bumps=args.num_bumps
    )

    # --- 3. 비디오 녹화 설정 ---
    if args.record:
        model_name = os.path.splitext(os.path.basename(args.model))[0]
        video_folder = f"./videos/{model_name}/"
        os.makedirs(video_folder, exist_ok=True)
        
        # RecordVideo 래퍼를 환경에 씌웁니다.
        # 이제 env의 render_mode는 'rgb_array'이므로 오류가 발생하지 않습니다.
        env = RecordVideo(
            env,
            video_folder=video_folder,
            name_prefix=f"render-{model_name}",
            episode_trigger=lambda x: True
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
    vec_env = model.get_env()
    obs = vec_env.reset()
    
    total_rewards = 0
    
    print("\n렌더링을 시작합니다. 종료하려면 Ctrl+C를 누르세요.")
    try:
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = vec_env.step(action)
            
            # --record 플래그가 없을 때만(human 모드일 때만) 직접 렌더링을 호출합니다.
            if not args.record:
                vec_env.render()

            total_rewards += rewards[0]

            if dones[0]:
                cleared_bumps = infos[0].get('cleared_bumps', 0)
                print(f"에피소드 종료. 총 보상: {total_rewards:.2f}, 통과한 장애물 수: {cleared_bumps}")
                total_rewards = 0

    except KeyboardInterrupt:
        print("\n사용자에 의해 렌더링이 중단되었습니다.")
    finally:
        # --- 6. 종료 처리 ---
        vec_env.close()
        print("환경이 종료되었습니다.")

if __name__ == "__main__":
    main()
