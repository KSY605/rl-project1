import argparse
import cv2
import numpy as np
import os
from stable_baselines3 import PPO
from custom_walker2d_origin import CustomEnvWrapper
import gymnasium as gym
import time # 파일명에 시간을 추가하기 위해 import

# --- (argparse 부분은 동일) ---
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default=None, help="Path to the saved model (.zip)")
parser.add_argument("--bump_practice", action="store_true", help="Enable bumping")
parser.add_argument("--bump_challenge", action="store_true", help="Enable bumping")
parser.add_argument("--record", action="store_true", help="Enable recording with R key toggle")
args = parser.parse_args()


# ▼▼▼ 영상 저장 로직을 함수로 분리 ▼▼▼
def save_video(frames_to_save, width, height, model_path):
    """녹화된 프레임들을 mp4 파일로 저장하는 함수"""
    if not frames_to_save:
        print("저장할 프레임이 없습니다.")
        return

    # 모델 경로가 없을 경우를 대비해 파일명 생성 로직 수정
    if model_path:
        base_name = os.path.splitext(os.path.basename(model_path))[0]
    else:
        base_name = "random_agent"
    
    # 파일명이 겹치지 않도록 현재 시간을 이용
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    file_name = f"recorded_{base_name}_{timestamp}.mp4"

    # 비디오 라이터 생성
    video_writer = cv2.VideoWriter(file_name,
                                   cv2.VideoWriter_fourcc(*'mp4v'),
                                   60, (width, height))
    # 프레임 저장
    for f in frames_to_save:
        video_writer.write(f)

    video_writer.release()
    print(f"'{file_name}'으로 영상이 성공적으로 저장되었습니다.")


render_mode = "rgb_array" if args.record else "human"
env = CustomEnvWrapper(render_mode=render_mode, bump_practice=args.bump_practice, bump_challenge=args.bump_challenge)
model = PPO.load(args.model) if args.model is not None else None
obs, _ = env.reset()

recording = False
frames = []

if args.record:
    print("녹화 기능 활성화. 'R' 키로 녹화 시작/중단, 'Q' 키로 종료.")

# 메인 루프
try: # 사용자가 창을 그냥 닫을 경우를 대비해 try-finally 사용
    while True:
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)

        if args.record:
            frame = env.render()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if recording:
                frames.append(frame_bgr)

            cv2.imshow("Walker2D", frame_bgr)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('r'):
                if not recording:
                    print("녹화를 시작합니다...")
                    recording = True
                    frames = [] # 새 녹화를 위해 프레임 리스트 초기화
                else:
                    print("녹화를 중단하고 저장을 시작합니다...")
                    recording = False
                    if frames:
                        height, width, _ = frames[0].shape
                        # ▼▼▼ 함수 호출로 변경 ▼▼▼
                        save_video(frames, width, height, args.model)

            # ▼▼▼ 종료(q) 시 저장 로직 추가 ▼▼▼
            elif key == ord('q'):
                print("프로그램을 종료합니다...")
                # 종료 시점에도 녹화 중이었다면 저장
                if recording and frames:
                    print("진행 중이던 녹화를 저장합니다...")
                    height, width, _ = frames[0].shape
                    save_video(frames, width, height, args.model)
                break

        if terminated or truncated:
            obs, _ = env.reset()

finally: # 프로그램이 어떤 이유로든 종료될 때 창을 확실히 닫음
    env.close()
    cv2.destroyAllWindows()