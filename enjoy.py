import os
import sys
import argparse
from stable_baselines3 import PPO

# custom_walker2d.py가 있는 디렉토리를 파이썬 경로에 추가합니다.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from custom_walker2d_v7 import CustomEnvWrapper 

def main():
    parser = argparse.ArgumentParser(description="학습된 Walker2D 모델을 렌더링합니다.")
    # 렌더링할 모델의 .zip 파일 경로를 인자로 받습니다.
    parser.add_argument("--model-path", type=str, required=True, help="렌더링할 모델 파일의 경로. (예: checkpoints/walker_finetuned/final_finetuned_model.zip)")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"오류: 모델 파일을 찾을 수 없습니다. 경로를 확인해주세요: {args.model_path}")
        return

    # --- 환경 생성 ---
    # 렌더링을 위해 render_mode="human"으로 환경을 생성합니다.
    # 학습 시와 동일한 환경을 사용해야 합니다.
    env = CustomEnvWrapper(render_mode="human", bump_challenge=True)

    # --- 모델 불러오기 ---
    print(f"모델을 불러옵니다: {args.model_path}")
    try:
        model = PPO.load(args.model_path, env=env)
    except Exception as e:
        print(f"모델을 불러오는 중 오류가 발생했습니다: {e}")
        print("모델을 저장할 때와 동일한 버전의 라이브러리(Stable-Baselines3, PyTorch)를 사용하고 있는지 확인해주세요.")
        env.close()
        return

    # --- 렌더링 루프 ---
    episodes = 10 # 10번의 에피소드를 실행합니다.
    for ep in range(episodes):
        obs, info = env.reset()
        terminated = False
        truncated = False
        total_reward = 0
        cleared_bumps = 0
        
        print(f"\n--- 에피소드 {ep + 1} 시작 ---")
        
        while not terminated and not truncated:
            # deterministic=True: 학습된 정책을 가장 확실하게 실행합니다.
            action, _states = model.predict(obs, deterministic=True)
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            
            # info 딕셔너리에 'event' 키가 있고 값이 'BUMP_CLEARED'일 때 카운트
            if info.get('event') == 'BUMP_CLEARED':
                cleared_bumps += 1
                print(f"Bump 통과! (총 {cleared_bumps}개)")

        print(f"에피소드 종료. 총 보상: {total_reward:.2f}, 통과한 Bump 개수: {cleared_bumps}")

    env.close()
    print("\n렌더링이 모두 종료되었습니다.")

if __name__ == "__main__":
    main()
