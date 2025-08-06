import gymnasium as gym
from gymnasium import spaces
import numpy as np
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import get_device
from collections import deque
import xml.etree.ElementTree as ET
import os
import tempfile
import argparse
import torch as th
import zipfile
# --- 오류 수정을 위해 직접 클래스를 임포트 ---
from gymnasium.envs.mujoco.walker2d_v5 import Walker2dEnv

# --- 0. TensorBoard 로깅을 위한 커스텀 콜백 ---
class TensorboardLoggingCallback(BaseCallback):
    """
    에피소드 종료 시 커스텀 정보를 TensorBoard에 로깅하는 콜백.
    """
    def __init__(self, verbose=0):
        super(TensorboardLoggingCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        for i, done in enumerate(self.locals['dones']):
            if done:
                info = self.locals['infos'][i]
                if 'passed_bumps_count' in info:
                    self.logger.record('custom/passed_bumps_count', info['passed_bumps_count'])
                if 'total_bumps_in_level' in info:
                    self.logger.record('custom/total_bumps_in_level', info['total_bumps_in_level'])
        return True

# --- 1. 커리큘럼 관리자 (Curriculum Manager) ---
class CurriculumManager:
    def __init__(self, bumps_xml_path, level_grouping_thresholds):
        self.all_bumps = self._parse_bumps(bumps_xml_path)
        self.levels = self._create_levels(level_grouping_thresholds)
        self.current_level = 0
        print(f"총 {len(self.levels)}개의 커리큘럼 레벨이 생성되었습니다.")
        for i, level_bumps in enumerate(self.levels):
            print(f"  - 레벨 {i}: 장애물 {len(level_bumps)}개")

    def _parse_bumps(self, xml_path):
        """XML 파일에서 장애물 정보를 파싱하고 x 위치 순으로 정렬합니다."""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        bumps = []
        for geom in root.findall(".//geom"):
            name = geom.get('name')
            if name and name.startswith('bump'):
                pos = [float(p) for p in geom.get('pos').split()]
                size = [float(s) for s in geom.get('size').split()]
                # XML에서 높이는 size의 z값(세 번째 요소)입니다.
                bumps.append({'name': name, 'pos_x': pos[0], 'height': size[2], 'xml_element': geom})
        return sorted(bumps, key=lambda b: b['pos_x'])

    def _create_levels(self, thresholds):
        """
        점진적으로 어려워지는 장애물 커리큘럼을 생성하며,
        최종 레벨이 고유하도록 보장합니다.
        """
        all_bumps_sorted_by_height = sorted(self.all_bumps, key=lambda b: b['height'])
        
        levels = []
        # 레벨 0: 장애물 없음. 기본적인 보행을 학습하기 위한 좋은 시작점입니다.
        levels.append([])
        
        # 중간 레벨: 각 높이 임계값까지의 장애물을 포함합니다.
        for threshold in sorted(thresholds):
            level_bumps = [b for b in all_bumps_sorted_by_height if b['height'] <= threshold]
            if level_bumps: # 임계값이 너무 낮아 빈 리스트가 추가되는 것을 방지합니다.
                levels.append(level_bumps)
                
        # 최종 레벨: 모든 장애물의 전체 집합을 추가합니다.
        levels.append(self.all_bumps)
        
        # 생성될 수 있는 중복 레벨을 제거합니다.
        unique_levels = []
        seen_levels_signatures = set()
        for level in levels:
            # 장애물 이름을 기반으로 레벨의 고유한 "시그니처"를 생성합니다.
            level_signature = tuple(sorted([b['name'] for b in level]))
            if level_signature not in seen_levels_signatures:
                unique_levels.append(level)
                seen_levels_signatures.add(level_signature)
                
        return unique_levels

    def get_current_level_bumps(self):
        return self.levels[self.current_level]

    def promote(self):
        if self.current_level < len(self.levels) - 1:
            self.current_level += 1
            print(f"\n🎉🎉🎉 축하합니다! 레벨 {self.current_level}로 승급했습니다! 🎉🎉🎉")
            print(f"이제 {len(self.get_current_level_bumps())}개의 장애물에 도전합니다.")
            return True
        else:
            print("\n🏆 모든 커리큘럼을 마스터했습니다! 최종 훈련을 시작합니다.")
            return False

# --- 2. 커스텀 워커 환경 (Custom Walker Environment) ---
class CurriculumWalkerEnv(gym.Wrapper):
    def __init__(self, base_xml_path, bumps_for_level, render_mode=None):
        self.base_xml_path = base_xml_path
        self.bumps_for_level = sorted(bumps_for_level, key=lambda b: b['pos_x'])
        self.temp_xml_path = self._create_temp_xml_with_bumps()
        
        env = Walker2dEnv(
            xml_file=self.temp_xml_path, 
            render_mode=render_mode,
            frame_skip=10, 
            healthy_z_range=(0.5, 10.0),
            exclude_current_positions_from_observation=False
        )
        super().__init__(env)
        
        self.passed_bumps = set()
        
        # 에이전트가 다음 N개의 장애물을 보도록 관찰 공간을 확장합니다.
        self.num_bumps_to_observe = 3 # 관찰할 장애물 수
        self.num_bump_features = 2 # 장애물 당 특징 수 (거리, 높이)
        base_obs_space = self.env.observation_space
        
        # 추가될 장애물 관찰 공간 정의
        bump_obs_low = np.full(self.num_bumps_to_observe * self.num_bump_features, -np.inf)
        bump_obs_high = np.full(self.num_bumps_to_observe * self.num_bump_features, np.inf)
        
        # 기존 관찰 공간과 장애물 관찰 공간을 합칩니다.
        new_low = np.concatenate([base_obs_space.low, bump_obs_low])
        new_high = np.concatenate([base_obs_space.high, bump_obs_high])
        self.observation_space = spaces.Box(low=new_low, high=new_high, dtype=np.float64)
        
    def _create_temp_xml_with_bumps(self):
        """현재 레벨에 맞는 장애물만 포함하는 임시 XML 파일을 생성합니다."""
        tree = ET.parse(self.base_xml_path)
        root = tree.getroot()
        worldbody = root.find('worldbody')
        
        # 기존의 모든 'bump' 지오메트리를 제거합니다.
        bumps_to_remove = [geom for geom in worldbody.findall(".//geom") if geom.get('name', '').startswith('bump')]
        for geom in bumps_to_remove:
            worldbody.remove(geom)

        # 현재 레벨에 해당하는 장애물만 추가합니다.
        for bump_info in self.bumps_for_level:
            worldbody.append(bump_info['xml_element'])
            
        # 임시 파일로 저장합니다.
        temp_dir = tempfile.gettempdir()
        # 프로세스 ID를 포함하여 병렬 실행 시 파일 충돌을 방지합니다.
        temp_path = os.path.join(temp_dir, f"walker_curriculum_{os.getpid()}.xml")
        tree.write(temp_path)
        return temp_path

    def _get_bump_observation(self):
        """에이전트의 현재 위치를 기준으로 다가오는 장애물 정보를 반환합니다."""
        agent_x = self.unwrapped.data.qpos[0]
        # 아직 지나가지 않은 장애물 목록
        upcoming_bumps = [b for b in self.bumps_for_level if b['pos_x'] > agent_x]
        
        bump_features = []
        for i in range(self.num_bumps_to_observe):
            if i < len(upcoming_bumps):
                bump = upcoming_bumps[i]
                # (에이전트로부터의 상대 거리, 장애물 높이)
                bump_features.extend([bump['pos_x'] - agent_x, bump['height']])
            else:
                # 관찰할 장애물이 더 이상 없으면 먼 거리와 0 높이를 나타내는 값을 채웁니다.
                bump_features.extend([100.0, 0.0]) # 큰 값으로 장애물이 없음을 표시
        return np.array(bump_features, dtype=np.float32)

    def _get_full_observation(self, base_obs):
        """기본 관찰값과 장애물 관찰값을 결합합니다."""
        return np.concatenate([base_obs, self._get_bump_observation()])

    def step(self, action):
        base_obs, reward, terminated, truncated, info = self.env.step(action)
        x_pos = self.unwrapped.data.qpos[0]
        
        # 통과한 장애물 기록
        for bump in self.bumps_for_level:
            if x_pos > bump['pos_x']:
                self.passed_bumps.add(bump['name'])
                
        info['passed_bumps_count'] = len(self.passed_bumps)
        info['total_bumps_in_level'] = len(self.bumps_for_level)
        
        # 현재 레벨의 모든 장애물을 통과했는지 확인
        if info['total_bumps_in_level'] > 0 and info['passed_bumps_count'] == info['total_bumps_in_level']:
            info['cleared_all_bumps'] = True
            reward += 100  # 큰 보상
            terminated = True # 에피소드 성공적으로 종료
        else:
            info['cleared_all_bumps'] = False
            
        return self._get_full_observation(base_obs), reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.passed_bumps.clear()
        base_obs, info = self.env.reset(**kwargs)
        info['passed_bumps_count'] = 0
        info['total_bumps_in_level'] = len(self.bumps_for_level)
        return self._get_full_observation(base_obs), info
        
    def close(self):
        super().close()
        # 환경이 닫힐 때 임시 XML 파일을 삭제합니다.
        if hasattr(self, 'temp_xml_path') and os.path.exists(self.temp_xml_path):
            os.remove(self.temp_xml_path)

# --- 3. 가중치 이식 함수 ---
def transfer_weights(pretrained_params, new_model_params):
    """Observation Space가 다른 모델 간 가중치를 이식합니다."""
    for (new_name, new_param), (old_name, old_param) in zip(new_model_params.items(), pretrained_params.items()):
        if new_param.shape == old_param.shape:
            new_param.data.copy_(old_param.data)
        # 입력 레이어(첫 번째 레이어)의 가중치를 부분적으로 복사
        elif 'policy_net.0.weight' in new_name or 'value_net.0.weight' in new_name:
            if len(new_param.shape) > 1 and len(old_param.shape) > 1:
                old_obs_dim = old_param.shape[1]
                # 새로운 모델의 가중치 텐서에서, 기존 observation에 해당하는 부분만 복사
                new_param.data[:, :old_obs_dim] = old_param.data
                print(f"입력 레이어 가중치 부분 복사 완료: {new_name}")
        else:
            print(f"레이어 크기 불일치로 건너뜀: {new_name}. New: {new_param.shape}, Old: {old_param.shape}")
    return new_model_params

# --- 4. 메인 훈련 루프 ---
def main(args):
    # --- 설정 ---
    XML_PATH = 'custom_walker2d_bumps_v2.xml'
    LEVEL_THRESHOLDS = [0.15, 0.3, 0.5, 0.8, 1.3] 
    TOTAL_TIMESTEPS = 2_000_000
    EVAL_EPISODES = 100
    PROMOTION_THRESHOLD = 0.95
    LEARNING_STEPS_PER_EVAL = 50_000
    
    # --- 초기화 ---
    CHECKPOINT_DIR = "checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    curriculum_manager = CurriculumManager(XML_PATH, LEVEL_THRESHOLDS)
    callback = TensorboardLoggingCallback()

    def make_env(rank=0, seed=0):
        def _init():
            bumps = curriculum_manager.get_current_level_bumps()
            env = CurriculumWalkerEnv(XML_PATH, bumps)
            env.reset(seed=seed + rank)
            return env
        return _init

    vec_env = SubprocVecEnv([make_env(rank=i) for i in range(args.n_envs)])
    
    policy_kwargs = {"net_arch": {"pi": [128, 64, 64], "vf": [128, 64, 64]}}
    
    # --- 모델 생성 또는 로드 ---
    if args.pretrained_model and os.path.exists(args.pretrained_model):
        print(f"사전 훈련된 모델 '{args.pretrained_model}'에서 가중치를 로드합니다.")
        # 먼저 새 환경에 맞는 모델 구조를 생성합니다.
        model = PPO("MlpPolicy", vec_env, verbose=0, tensorboard_log="./walker_tensorboard/", policy_kwargs=policy_kwargs)
        
        # 사전 훈련된 모델의 가중치만 불러옵니다.
        with zipfile.ZipFile(args.pretrained_model, "r") as archive:
            # policy.pth 파일 찾기
            policy_file_path = next((name for name in archive.namelist() if name.endswith("policy.pth")), None)
            if policy_file_path is None:
                raise FileNotFoundError(f"'{args.pretrained_model}' zip 파일에서 policy.pth 파일을 찾을 수 없습니다.")

            with archive.open(policy_file_path, "r") as policy_file:
                pretrained_params = th.load(policy_file, map_location=get_device("auto"))

        new_model_params = model.policy.state_dict()
        # 가중치 이식
        transfer_weights(pretrained_params, new_model_params)
        model.policy.load_state_dict(new_model_params)
        print("가중치 이식 완료.")
    else:
        print("새로운 PPO 모델을 생성합니다.")
        model = PPO("MlpPolicy", vec_env, verbose=0, tensorboard_log="./walker_tensorboard/", policy_kwargs=policy_kwargs)

    # --- 훈련 시작 ---
    timesteps_done = 0
    while timesteps_done < TOTAL_TIMESTEPS:
        # reset_num_timesteps=False로 설정하여 총 타임스텝이 누적되도록 합니다.
        model.learn(total_timesteps=LEARNING_STEPS_PER_EVAL, reset_num_timesteps=False, 
                    tb_log_name=f"PPO_Level_{curriculum_manager.current_level}", callback=callback)
        timesteps_done += LEARNING_STEPS_PER_EVAL
        print(f"\n--- 총 진행률: {timesteps_done}/{TOTAL_TIMESTEPS} 타임스텝 ---")
        print(f"현재 레벨 {curriculum_manager.current_level}에서 성능 평가를 시작합니다...")

        # 평가용 환경 생성
        eval_env = DummyVecEnv([make_env()])
        successful_episodes = 0
        for _ in range(EVAL_EPISODES):
            obs = eval_env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                # VecEnv의 step 메소드는 4개의 값을 반환합니다: (obs, rewards, dones, infos)
                obs, rewards, dones, infos = eval_env.step(action)
                
                # VecEnv는 여러 환경을 다루므로 결과는 배열/리스트 형태입니다.
                # 여기서는 환경이 하나이므로 첫 번째 요소에 접근합니다.
                done = dones[0]
                info = infos[0]

                # --- 🔴 에러 수정 🔴 ---
                # 에피소드가 끝났을 때 성공 여부를 판단합니다.
                if done:
                    # 레벨 0 (장애물 없음)의 경우, 에피소드가 끝나면(넘어지지 않으면) 성공으로 간주합니다.
                    if info.get('total_bumps_in_level', 0) == 0:
                        successful_episodes += 1
                        break # while 루프 탈출
                    # 장애물이 있는 레벨의 경우, 모든 장애물을 통과해야 성공입니다.
                    elif info.get('cleared_all_bumps', False):
                        successful_episodes += 1
                        # break는 필요 없습니다. cleared_all_bumps가 True이면 어차피 terminated=True가 되어 done이 True가 됩니다.

        eval_env.close()
        
        success_rate = successful_episodes / EVAL_EPISODES
        print(f"평가 완료: 성공률 {success_rate:.2%} ({successful_episodes}/{EVAL_EPISODES})")

        # 성공률이 임계값을 넘으면 다음 레벨로 진행
        if success_rate >= PROMOTION_THRESHOLD:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"walker_level_{curriculum_manager.current_level}_ec_best.zip")
            model.save(checkpoint_path)
            print(f"✅ 체크포인트 저장 완료: {checkpoint_path}")

            # 다음 레벨로 승급. 더 이상 레벨이 없으면 루프 종료
            if not curriculum_manager.promote():
                break 
            
            # 다음 레벨을 위해 환경을 새로 만듭니다.
            vec_env.close()
            vec_env = SubprocVecEnv([make_env(rank=i) for i in range(args.n_envs)])
            model.set_env(vec_env)
        else:
            print(f"성공률이 목표({PROMOTION_THRESHOLD:.0%})에 도달하지 못했습니다. 현재 레벨에서 훈련을 계속합니다.")

    print("\n--- 최종 훈련 단계 ---")
    # 남은 타임스텝만큼 최종 훈련을 진행합니다.
    model.learn(total_timesteps=max(0, TOTAL_TIMESTEPS - timesteps_done), reset_num_timesteps=False, 
                tb_log_name="PPO_Final", callback=callback)
    
    model.save("walker_curriculum_final.zip")
    print("최종 모델이 'walker_curriculum_final.zip'으로 저장되었습니다.")
    vec_env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Curriculum Learning for Walker2D with Bumps")
    parser.add_argument("--pretrained-model", type=str, default=None, help="사전 훈련된 모델의 경로 (예: ./walker.zip)")
    parser.add_argument("--n-envs", type=int, default=4, help="병렬로 실행할 환경의 수 (CPU 코어 수에 맞게 조절)")
    cli_args = parser.parse_args()

    if not os.path.exists('custom_walker2d_bumps_v2.xml'):
        print("에러: 'custom_walker2d_bumps_v2.xml' 파일을 찾을 수 없습니다.")
    else:
        main(cli_args)
