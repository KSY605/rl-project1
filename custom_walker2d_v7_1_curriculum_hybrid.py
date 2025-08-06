import numpy as np
import gymnasium as gym
import os
from typing import List, Tuple, Set

class CustomEnvWrapper(gym.Wrapper):
    """
    Walker2d 환경에 하이브리드 커리큘럼 및 단계별 보상 설계를 적용한 래퍼입니다.
    """

    def __init__(self, render_mode=None, bump_practice=False, bump_challenge=False, curriculum_level: int = 0, num_bumps: int = 0):
        # --- 1. 환경 로드 (이전과 동일) ---
        repo_root = os.path.dirname(os.path.abspath(__file__))
        asset_dir = os.path.join(repo_root, "asset")
        
        self.bump_challenge = bump_challenge
        self.curriculum_level = curriculum_level
        xml_file = None

        if self.curriculum_level > 0:
            xml_file_path = os.path.join(asset_dir, f"curriculum_level{self.curriculum_level}.xml")
            if os.path.exists(xml_file_path):
                xml_file = xml_file_path
                print(f"--- Loading Curriculum Level {self.curriculum_level}: {os.path.basename(xml_file)} ---")
            else:
                print(f"--- WARNING: curriculum_level{self.curriculum_level}.xml not found. Using default. ---")
        elif self.bump_challenge:
            xml_file = os.path.join(asset_dir, "custom_walker2d_bumps_v2.xml")
            print(f"--- Loading Final Challenge: custom_walker2d_bumps_v2.xml ---")
        elif bump_practice:
            xml_file = os.path.join(asset_dir, "custom_walker2d_bumps_practice_v2.xml")
        
        env = gym.make(
            "Walker2d-v5", xml_file=xml_file, render_mode=render_mode,
            exclude_current_positions_from_observation=False, frame_skip=10, healthy_z_range=(0.4, 10.0),
        )
        super().__init__(env)

        self.base_env = env.unwrapped
        base_model = self.base_env.model
        
        all_bump_geoms = sorted(
            [(i, base_model.geom(i).name) for i in range(base_model.ngeom)
             if base_model.geom(i).name and base_model.geom(i).name.startswith("bump")],
            key=lambda item: int(item[1].replace("bump", "")))
        all_bump_geom_ids = [item[0] for item in all_bump_geoms]
        
        self.active_bump_geom_ids = all_bump_geom_ids
        if self.bump_challenge and num_bumps > 0:
            self.active_bump_geom_ids = all_bump_geom_ids[:num_bumps]
            print(f"--- Challenge Mode: Activating first {len(self.active_bump_geom_ids)} bumps. ---")

        self.foot_geom_ids = {base_model.geom(name).id for name in ["foot_geom", "foot_left_geom"]}
        
        # --- 2. 상태 변수 초기화 ---
        self.cleared_bumps_count = 0
        self.current_bump_target_x = np.inf
        self.steps_since_clear = 0 # 안정적인 착지 보상을 위한 카운터

        sample_obs, _ = self.reset()
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=sample_obs.shape, dtype=np.float64)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.cleared_bumps_count = 0
        self.steps_since_clear = 0 # 리셋 시 카운터 초기화
        
        next_bump_info = self._get_next_n_bumps_info(n=1)
        if next_bump_info and next_bump_info[0][3] != -1.0:
            self.current_bump_target_x = next_bump_info[0][3]
        else:
            self.current_bump_target_x = np.inf
        return self.custom_observation(obs), info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        
        custom_obs = self.custom_observation(obs)
        custom_reward, success_bonus = self._calculate_rewards(obs, custom_obs, action)
        
        # 장애물 통과 시 카운터 리셋
        if success_bonus > 0:
            self.steps_since_clear = 0
        else:
            self.steps_since_clear += 1

        info['cleared_bumps'] = self.cleared_bumps_count
        if success_bonus > 0: info['event'] = 'BUMP_CLEARED'
            
        return custom_obs, custom_reward, terminated, truncated, info

    def _calculate_rewards(self, obs: np.ndarray, custom_obs: np.ndarray, action: np.ndarray) -> Tuple[float, float]:
        """
        커리큘럼 단계에 따라 보상 함수를 동적으로 계산합니다.
        """
        # --- 관측값 및 파라미터 ---
        z_torso, vel_x, angvel_torso = obs[1], obs[9], obs[11]
        bump1_dx, bump1_h, bump1_w = custom_obs[17], custom_obs[18], custom_obs[19]
        is_over_bump = (bump1_dx >= -bump1_w) and (bump1_dx <= bump1_w)

        # --- 1. 기반 보상 (모든 단계 공통) ---
        W_FORWARD, W_ALIVE, W_CTRL, W_FALL_PENALTY = 2.0, 0.1, -0.01, -50.0
        # ==================== 수정된 부분 ====================
        # 안정성 페널티를 기반 보상에 포함시켜 모든 레벨에 적용합니다.
        W_STABILITY_PENALTY = -0.001
        
        reward_forward = W_FORWARD * vel_x
        reward_alive = W_ALIVE        
        penalty_control = W_CTRL * np.sum(np.square(action))
        penalty_stability = W_STABILITY_PENALTY * np.square(angvel_torso) # 몸이 크게 기울어지면 페널티
        penalty_fall = W_FALL_PENALTY if z_torso < 0.7 else 0.0        
        base_reward = reward_forward + reward_alive + penalty_control + penalty_stability + penalty_fall
        # ====================================================

        # --- 2. 단계별 추가 보상 ---
        reward_clearance = 0.0
        reward_stability = 0.0
        reward_preparation = 0.0

        # Level 2 이상: '최소 높이 확보' 보상
        if self.curriculum_level >= 2 or self.bump_challenge:
            W_CLEARANCE, CLEARANCE_MARGIN = 25.0, 0.05
            if is_over_bump:
                clearance = z_torso - (bump1_h * 2 + CLEARANCE_MARGIN)
                reward_clearance = W_CLEARANCE * max(0, clearance)

        # Level 3 이상: '안정적인 착지' 보상
        if self.curriculum_level >= 3 or self.bump_challenge:
            W_STABILITY, STABILITY_WINDOW = 8.0, 15 # 통과 후 15스텝 동안
            if self.steps_since_clear < STABILITY_WINDOW:
                stability_bonus = np.exp(-2.0 * abs(angvel_torso)) # 각속도가 0에 가까울수록 보상 증가
                reward_stability = W_STABILITY * stability_bonus

        # Level 4 이상: '준비 자세' 보상
        if self.curriculum_level >= 4 or self.bump_challenge:
            W_CROUCH, CROUCH_TARGET_Z, PREP_DIST = 10.0, 0.9, 2.0
            is_approaching_high_bump = (bump1_dx > 0 and bump1_dx < PREP_DIST and bump1_h > 0.5)
            if is_approaching_high_bump:
                crouch_bonus = np.exp(-5.0 * (z_torso - CROUCH_TARGET_Z)**2) # 목표 높이에 가까울수록 보상 증가
                reward_preparation = W_CROUCH * crouch_bonus
        
        # --- 3. 장애물 통과 성공 보너스 (공통) ---
        SUCCESS_BONUS, MIN_SUCCESS_HEIGHT = 50.0, 1.0
        success_bonus = 0.0
        next_bump_info = self._get_next_n_bumps_info(n=1)
        next_bump_x_pos = next_bump_info[0][3] if next_bump_info else -1.0
        
        if self.current_bump_target_x > 0 and self.current_bump_target_x != next_bump_x_pos and z_torso > MIN_SUCCESS_HEIGHT:
            success_bonus = SUCCESS_BONUS
            self.cleared_bumps_count += 1
        self.current_bump_target_x = next_bump_x_pos

        # --- 최종 보상 합산 ---
        total_reward = (base_reward + 
                        reward_clearance + 
                        reward_stability + 
                        reward_preparation + 
                        success_bonus)
                        
        return total_reward, success_bonus

    # _get_next_n_bumps_info, custom_observation 등 나머지 함수는 이전과 동일
    # ... (이전 코드와 동일) ...
    def _is_foot_on_bump(self, target_bump_name: str) -> bool:
        try:
            target_bump_id = self.base_env.model.geom(target_bump_name).id
        except KeyError: return False
        for i in range(self.base_env.data.ncon):
            contact = self.base_env.data.contact[i]
            geom1, geom2 = contact.geom1, contact.geom2
            is_foot_contact = (geom1 in self.foot_geom_ids and geom2 == target_bump_id) or \
                              (geom2 in self.foot_geom_ids and geom1 == target_bump_id)
            if is_foot_contact: return True
        return False

    def _get_next_n_bumps_info(self, n: int = 4) -> List[Tuple[float, float, float, float]]:
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
