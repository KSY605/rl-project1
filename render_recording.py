import os
import sys
import argparse
import gymnasium as gym
from stable_baselines3 import PPO

# custom_walker2d.py가 있는 디렉토리를 파이썬 경로에 추가합니다.
# Add the directory containing custom_walker2d.py to the Python path.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from custom_walker2d_v2 import CustomEnvWrapper 

def main():
    """
    학습된 모델을 렌더링하고 선택적으로 비디오를 녹화하는 메인 함수.
    Main function to render a trained model and optionally record a video.
    """
    parser = argparse.ArgumentParser(description="학습된 Walker2D 모델을 렌더링하고 녹화합니다.")
    # 렌더링할 모델의 .zip 파일 경로를 인자로 받습니다.
    # Argument for the path to the .zip file of the model to be rendered.
    parser.add_argument("--model-path", type=str, required=True, help="렌더링할 모델 파일의 경로. (예: checkpoints/walker_finetuned/final_finetuned_model.zip)")
    # 비디오 녹화 여부를 결정하는 인자입니다.
    # Argument to decide whether to record a video.
    parser.add_argument("--record", action="store_true", help="렌더링 결과를 비디오로 녹화하려면 이 플래그를 추가하세요.")
    # 녹화된 비디오를 저장할 폴더를 지정하는 인자입니다.
    # Argument to specify the folder for saving the recorded video.
    parser.add_argument("--video-folder", type=str, default="videos", help="녹화된 비디오를 저장할 폴더 경로.")
    
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"오류: 모델 파일을 찾을 수 없습니다. 경로를 확인해주세요: {args.model_path}")
        return

    # --- 환경 생성 (Environment Creation) ---
    # 녹화 시에는 "rgb_array", 렌더링만 할 때는 "human" 모드를 사용합니다.
    # Use "rgb_array" for recording, and "human" for rendering only.
    render_mode = "rgb_array" if args.record else "human"
    env = CustomEnvWrapper(render_mode=render_mode, bump_challenge=True)

    # --- 비디오 녹화 래퍼 적용 (Apply Video Recording Wrapper) ---
    if args.record:
        print(f"비디오 녹화를 활성화합니다. 저장 폴더: {args.video_folder}")
        print("참고: 녹화 중에는 렌더링 창이 화면에 나타나지 않습니다.")
        # 지정된 폴더가 없으면 생성합니다.
        # Create the folder if it doesn't exist.
        os.makedirs(args.video_folder, exist_ok=True)
        # RecordVideo 래퍼로 환경을 감쌉니다.
        # Wrap the environment with the RecordVideo wrapper.
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=args.video_folder,
            name_prefix=f"ppo-walker-{os.path.basename(args.model_path).replace('.zip', '')}",
            # 매 5번째 에피소드만 녹화하려면 아래 주석을 해제하세요.
            # Uncomment the line below to record every 5th episode.
            # episode_trigger=lambda x: x % 5 == 0
        )


    # --- 모델 불러오기 (Load Model) ---
    print(f"모델을 불러옵니다: {args.model_path}")
    try:
        model = PPO.load(args.model_path, env=env)
    except Exception as e:
        print(f"모델을 불러오는 중 오류가 발생했습니다: {e}")
        print("모델을 저장할 때와 동일한 버전의 라이브러리(Stable-Baselines3, PyTorch)를 사용하고 있는지 확인해주세요.")
        env.close()
        return

    # --- 렌더링 루프 (Rendering Loop) ---
    episodes = 10 # 10번의 에피소드를 실행합니다.
    for ep in range(episodes):
        obs, info = env.reset()
        terminated = False
        truncated = False
        total_reward = 0
        cleared_bumps = 0
        
        print(f"\n--- 에피소드 {ep + 1} 시작 ---")
        
        while not terminated and not truncated:
            # RecordVideo 래퍼를 사용할 때는 env.render()를 명시적으로 호출할 필요가 없습니다.
            # When using the RecordVideo wrapper, you don't need to call env.render() explicitly.
            # step 함수 내부에서 자동으로 처리됩니다.
            # It's handled automatically inside the step function.
            
            # deterministic=True: 학습된 정책을 가장 확실하게 실행합니다.
            # deterministic=True: Executes the learned policy in the most deterministic way.
            action, _states = model.predict(obs, deterministic=True)
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            
            # info 딕셔너리에 'event' 키가 있고 값이 'BUMP_CLEARED'일 때 카운트
            # Count when the 'event' key in the info dictionary has the value 'BUMP_CLEARED'.
            if info.get('event') == 'BUMP_CLEARED':
                cleared_bumps += 1
                print(f"Bump 통과! (총 {cleared_bumps}개)")

        print(f"에피소드 종료. 총 보상: {total_reward:.2f}, 통과한 Bump 개수: {cleared_bumps}")

    # env.close()는 매우 중요합니다. 마지막 비디오 프레임을 저장하고 파일을 완성하는 역할을 합니다.
    # env.close() is very important. It saves the last video frame and finalizes the file.
    env.close()
    print("\n렌더링이 모두 종료되었습니다.")
    if args.record:
        print(f"녹화된 비디오는 '{args.video_folder}' 폴더에 저장되었습니다.")

if __name__ == "__main__":
    main()
