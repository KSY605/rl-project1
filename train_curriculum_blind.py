import gymnasium as gym
import numpy as np
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from collections import deque
import xml.etree.ElementTree as ET
import os
import tempfile

# --- 0. TensorBoard 로깅을 위한 커스텀 콜백 ---
class TensorboardLoggingCallback(BaseCallback):
    """
    에피소드 종료 시 커스텀 정보를 TensorBoard에 로깅하는 콜백.
    """
    def __init__(self, verbose=0):
        super(TensorboardLoggingCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # self.locals['dones']는 각 환경의 종료 여부를 담은 numpy 배열입니다.
        for i, done in enumerate(self.locals['dones']):
            if done:
                # 에피소드가 종료되었을 때만 정보를 로깅합니다.
                info = self.locals['infos'][i]
                
                if 'passed_bumps_count' in info:
                    passed_count = info['passed_bumps_count']
                    # 'custom' 섹션에 통과한 장애물 수를 기록합니다.
                    self.logger.record('custom/passed_bumps_count', passed_count)
                
                if 'total_bumps_in_level' in info:
                    total_count = info['total_bumps_in_level']
                    # 'custom' 섹션에 현재 레벨의 총 장애물 수를 기록합니다.
                    self.logger.record('custom/total_bumps_in_level', total_count)
        return True

# --- 1. 커리큘럼 관리자 (Curriculum Manager) ---
# 커리큘럼의 단계를 정의하고, 에이전트의 성공률을 추적하여 레벨 전환을 관리합니다.
class CurriculumManager:
    def __init__(self, bumps_xml_path, level_grouping_thresholds):
        """
        Args:
            bumps_xml_path (str): 장애물 정보가 담긴 MuJoCo XML 파일 경로
            level_grouping_thresholds (list): 장애물 높이에 따라 레벨을 나누는 기준 (e.g., [0.2, 0.5, 1.0])
        """
        self.all_bumps = self._parse_bumps(bumps_xml_path)
        self.levels = self._create_levels(level_grouping_thresholds)
        self.current_level = 0
        print(f"총 {len(self.levels)}개의 커리큘럼 레벨이 생성되었습니다.")
        for i, level_bumps in enumerate(self.levels):
            print(f"  - 레벨 {i}: 장애물 {len(level_bumps)}개")

    def _parse_bumps(self, xml_path):
        """XML 파일에서 장애물 정보를 파싱하고 높이 순으로 정렬합니다."""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        bumps = []
        for geom in root.findall(".//geom[starts-with(@name, 'bump')]"):
            pos_str = geom.get('pos')
            size_str = geom.get('size')
            pos = [float(p) for p in pos_str.split()]
            size = [float(s) for s in size_str.split()]
            bump_info = {
                'name': geom.get('name'),
                'pos_x': pos[0],
                'height': size[2],
                'xml_element': geom
            }
            bumps.append(bump_info)
        # 높이가 낮은 순서대로 정렬하여 커리큘럼을 구성
        return sorted(bumps, key=lambda b: b['height'])

    def _create_levels(self, thresholds):
        """정렬된 장애물을 높이 기준에 따라 레벨별로 그룹화합니다."""
        levels = []
        # Level 0: 장애물 없음
        levels.append([])
        
        # 높이 임계값에 따라 레벨 생성
        all_bumps_copy = self.all_bumps.copy()
        
        for threshold in thresholds:
            # 현재 임계값보다 낮은 모든 장애물을 포함
            level_bumps = [b for b in all_bumps_copy if b['height'] <= threshold]
            if level_bumps:
                levels.append(level_bumps)
        
        # 마지막 레벨: 모든 장애물 포함
        levels.append(all_bumps_copy)
        
        # 중복 레벨 제거
        unique_levels = []
        seen_levels = set()
        for level in levels:
            # 레벨을 식별하기 위해 장애물 이름의 튜플을 사용
            level_signature = tuple(sorted([b['name'] for b in level]))
            if level_signature not in seen_levels:
                unique_levels.append(level)
                seen_levels.add(level_signature)
        
        return unique_levels

    def get_current_level_bumps(self):
        """현재 커리큘럼 레벨에 해당하는 장애물 목록을 반환합니다."""
        return self.levels[self.current_level]

    def promote(self):
        """커리큘럼 레벨을 한 단계 올립니다."""
        if self.current_level < len(self.levels) - 1:
            self.current_level += 1
            print(f"\n🎉🎉🎉 축하합니다! 레벨 {self.current_level}로 승급했습니다! 🎉🎉🎉")
            print(f"이제 {len(self.get_current_level_bumps())}개의 장애물에 도전합니다.")
            return True
        else:
            print("\n🏆 모든 커리큘럼을 마스터했습니다! 최종 훈련을 시작합니다.")
            return False

# --- 2. 커스텀 워커 환경 (Custom Walker Environment) ---
# 동적으로 장애물을 조절하고, 성공 여부를 추적하는 커스텀 환경입니다.
class CurriculumWalkerEnv(gym.Wrapper):
    def __init__(self, base_xml_path, bumps_for_level):
        """
        Args:
            base_xml_path (str): 원본 MuJoCo XML 파일 경로
            bumps_for_level (list): 현재 레벨에 포함될 장애물 정보 목록
        """
        self.base_xml_path = base_xml_path
        self.bumps_for_level = bumps_for_level
        
        # 임시 XML 파일 생성
        self.temp_xml_path = self._create_temp_xml_with_bumps()
        
        # Gymnasium 환경 생성
        env = gym.make('Walker2d-v4', xml_file=self.temp_xml_path, render_mode=None)
        super().__init__(env)
        
        self.bump_positions_x = sorted([b['pos_x'] for b in self.bumps_for_level])
        self.passed_bumps = set()

    def _create_temp_xml_with_bumps(self):
        """현재 레벨의 장애물만 포함하는 임시 XML 파일을 생성합니다."""
        tree = ET.parse(self.base_xml_path)
        root = tree.getroot()
        worldbody = root.find('worldbody')

        # 기존 장애물 모두 제거
        for geom in worldbody.findall(".//geom[starts-with(@name, 'bump')]"):
            worldbody.remove(geom)
            
        # 현재 레벨의 장애물 추가
        for bump_info in self.bumps_for_level:
            worldbody.append(bump_info['xml_element'])
            
        # 임시 파일에 저장
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"walker_curriculum_{os.getpid()}.xml")
        tree.write(temp_path)
        return temp_path

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        x_position = self.unwrapped.data.qpos[0]
        
        for bump_x in self.bump_positions_x:
            if x_position > bump_x:
                self.passed_bumps.add(bump_x)
        
        info['passed_bumps_count'] = len(self.passed_bumps)
        info['total_bumps_in_level'] = len(self.bumps_for_level)
        
        if info['passed_bumps_count'] == info['total_bumps_in_level'] and info['total_bumps_in_level'] > 0:
            info['cleared_all_bumps'] = True
            reward += 100 
            terminated = True 
        else:
            info['cleared_all_bumps'] = False

        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.passed_bumps.clear()
        # reset()은 튜플을 반환해야 합니다. (obs, info)
        obs, info = self.env.reset(**kwargs)
        info['passed_bumps_count'] = 0
        info['total_bumps_in_level'] = len(self.bumps_for_level)
        return obs, info
        
    def close(self):
        """환경 종료 시 임시 XML 파일 삭제"""
        super().close()
        if os.path.exists(self.temp_xml_path):
            os.remove(self.temp_xml_path)


# --- 3. 메인 훈련 루프 (Main Training Loop) ---
def main():
    # --- 설정 ---
    XML_PATH = 'custom_walker2d_bumps_v2.xml'
    PRETRAINED_MODEL_PATH = None
    
    LEVEL_THRESHOLDS = [0.15, 0.3, 0.5, 0.8, 1.3] 
    
    TOTAL_TIMESTEPS = 2_000_000
    EVAL_EPISODES = 100
    PROMOTION_THRESHOLD = 0.95
    LEARNING_STEPS_PER_EVAL = 50_000
    
    # --- 초기화 ---
    CHECKPOINT_DIR = "checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    curriculum_manager = CurriculumManager(XML_PATH, LEVEL_THRESHOLDS)
    callback = TensorboardLoggingCallback() # 커스텀 콜백 인스턴스 생성

    def make_env():
        bumps = curriculum_manager.get_current_level_bumps()
        env = CurriculumWalkerEnv(XML_PATH, bumps)
        return env

    vec_env = DummyVecEnv([make_env])

    if PRETRAINED_MODEL_PATH and os.path.exists(PRETRAINED_MODEL_PATH):
        print(f"사전 훈련된 모델 '{PRETRAINED_MODEL_PATH}'을 로드합니다.")
        model = PPO.load(PRETRAINED_MODEL_PATH, env=vec_env)
    else:
        print("새로운 PPO 모델을 생성합니다.")
        model = PPO("MlpPolicy", vec_env, verbose=0, tensorboard_log="./walker_tensorboard/")

    # --- 훈련 시작 ---
    timesteps_done = 0
    while timesteps_done < TOTAL_TIMESTEPS:
        
        model.learn(total_timesteps=LEARNING_STEPS_PER_EVAL, reset_num_timesteps=False, 
                    tb_log_name=f"PPO_Level_{curriculum_manager.current_level}",
                    callback=callback) # <--- 여기에 콜백 추가
        timesteps_done += LEARNING_STEPS_PER_EVAL
        print(f"\n--- 총 진행률: {timesteps_done}/{TOTAL_TIMESTEPS} 타임스텝 ---")
        print(f"현재 레벨 {curriculum_manager.current_level}에서 성능 평가를 시작합니다...")

        successful_episodes = 0
        for _ in range(EVAL_EPISODES):
            obs, info = vec_env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = vec_env.step(action)
                done = terminated or truncated
                
                if done and info[0].get('cleared_all_bumps', False):
                    successful_episodes += 1
        
        success_rate = successful_episodes / EVAL_EPISODES
        print(f"평가 완료: 성공률 {success_rate:.2%} ({successful_episodes}/{EVAL_EPISODES})")

        if success_rate >= PROMOTION_THRESHOLD:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"walker_level_{curriculum_manager.current_level}_cleared.zip")
            model.save(checkpoint_path)
            print(f"✅ 체크포인트 저장 완료: {checkpoint_path}")

            if not curriculum_manager.promote():
                break 
            
            vec_env.close()
            vec_env = DummyVecEnv([make_env])
            model.set_env(vec_env)
        else:
            print(f"성공률이 목표({PROMOTION_THRESHOLD:.0%})에 도달하지 못했습니다. 현재 레벨에서 훈련을 계속합니다.")

    print("\n--- 최종 훈련 단계 ---")
    print("모든 커리큘럼을 통과했거나 최대 타임스텝에 도달했습니다.")
    model.learn(total_timesteps=max(0, TOTAL_TIMESTEPS - timesteps_done), reset_num_timesteps=False, 
                tb_log_name="PPO_Final", callback=callback) # <--- 여기에도 콜백 추가
    
    model.save("walker_curriculum_final.zip")
    print("최종 모델이 'walker_curriculum_final.zip'으로 저장되었습니다.")
    
    vec_env.close()


if __name__ == '__main__':
    if not os.path.exists('custom_walker2d_bumps_v2.xml'):
        print("에러: 'custom_walker2d_bumps_v2.xml' 파일을 찾을 수 없습니다.")
        print("스크립트와 동일한 디렉토리에 XML 파일을 위치시켜 주세요.")
    else:
        main()
