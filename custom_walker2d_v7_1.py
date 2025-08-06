import numpy as np
import gymnasium as gym
import os
from typing import List, Tuple, Set

class CustomEnvWrapper(gym.Wrapper):
    """
    Walker2d 환경에 지능적인 장애물 통과를 위한 커스텀 로직을 추가하는 래퍼입니다. (v8)

    - v8 변경점:
      - '장애물 통과 진행도'에 따른 조밀한 보상(Dense Reward) 로직 추가
      - 에이전트가 장애물 위에서 앞으로 나아가는 매 스텝마다 보상을 받아,
        보다 안정적이고 적극적인 장애물 돌파 행동을 학습하도록 유도
    """

    def __init__(self, render_mode=None, bump_practice=False, bump_challenge=False):
        # --- 환경 초기화 (기존과 동일) ---
        repo_root = os.path.dirname(os.path.abspath(__file__))
        asset_dir = os.path.join(repo_root, "asset")

        if bump_challenge:
            xml_file = os.path.join(asset_dir, "custom_walker2d_bumps_v2.xml")
        elif bump_practice:
            practice_xml_path = os.path.join(asset_dir, "custom_walker2d_bumps_practice_v2.xml")
            xml_file = practice_xml_path if os.path.exists(practice_xml_path) else os.path.join(asset_dir, "custom_walker2d_bumps_v2.xml")
        else:
            xml_file = None

        env = gym.make(
            "Walker2d-v5",
            xml_file=xml_file,
            render_mode=render_mode,
            exclude_current_positions_from_observation=False,
            frame_skip=10,
            healthy_z_range=(0.4, 10.0), 
        )
        super().__init__(env)

        self.base_env = env.unwrapped
        base_model = self.base_env.model
        
        self.bump_geom_ids = {
            i for i in range(base_model.ngeom)
            if base_model.geom(i).name and base_model.geom(i).name.startswith("bump")
        }
        self.foot_geom_ids = {
            base_model.geom(name).id for name in ["foot_geom", "foot_left_geom"]
        }
        
        # --- 변수 초기화 ---
        self.cleared_bumps_count = 0
        self.current_bump_target_x = np.inf
        self.prev_bump1_dx = np.inf
        self.last_progress_on_bump = 0.0 # v8 추가: 진행도 저장을 위한 변수

        sample_obs, _ = self.reset()
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=sample_obs.shape, dtype=np.float64
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.cleared_bumps_count = 0
        
        # ==================== v8 변경점: 진행도 변수 초기화 ====================
        self.last_progress_on_bump = 0.0
        # ====================================================================
        
        next_bump_info = self._get_next_n_bumps_info(n=1)
        if next_bump_info and next_bump_info[0][3] != -1.0:
            self.current_bump_target_x = next_bump_info[0][3]
            self.prev_bump1_dx = next_bump_info[0][0]
        else:
            self.current_bump_target_x = np.inf
            self.prev_bump1_dx = np.inf

        return self.custom_observation(obs), info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        
        custom_obs = self.custom_observation(obs)
        custom_reward, success_bonus = self._calculate_rewards(obs, custom_obs, action)
        
        if custom_reward < -40: 
            terminated = True

        info['cleared_bumps'] = self.cleared_bumps_count
        if success_bonus > 0:
            info['event'] = 'BUMP_CLEARED'
            
        return custom_obs, custom_reward, terminated, truncated, info

    def _is_foot_on_bump(self, target_bump_name: str) -> bool:
        """지정된 bump 이름에 발이 닿았는지 확인합니다. (기존과 동일)"""
        try:
            target_bump_id = self.base_env.model.geom(target_bump_name).id
        except KeyError:
            return False

        for i in range(self.base_env.data.ncon):
            contact = self.base_env.data.contact[i]
            geom1, geom2 = contact.geom1, contact.geom2

            is_foot_contact = (geom1 in self.foot_geom_ids and geom2 == target_bump_id) or \
                              (geom2 in self.foot_geom_ids and geom1 == target_bump_id)
            
            if is_foot_contact:
                return True
        return False

    def _get_next_n_bumps_info(self, n: int = 4) -> List[Tuple[float, float, float, float]]:
        """다음 n개의 장애물 정보를 가져옵니다. (기존과 동일)"""
        data, model = self.base_env.data, self.base_env.model
        walker_x = data.qpos[0]
        
        upcoming_bumps = []
        for gid in self.bump_geom_ids:
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
        """커스텀 관측값을 생성합니다. (기존과 동일)"""
        next_bumps_info = self._get_next_n_bumps_info(n=4)
        obs_bumps_info = [info[:3] for info in next_bumps_info]
        flat_bumps_info = np.array(obs_bumps_info, dtype=np.float64).flatten()
        return np.concatenate([obs, flat_bumps_info])
    
    def _calculate_rewards(self, obs: np.ndarray, custom_obs: np.ndarray, action: np.ndarray) -> Tuple[float, float]:
        # --- 하이퍼파라미터 ---
        W_FORWARD, W_ALIVE, W_CTRL, W_STABILITY = 1.2, 0.1, -0.01, -0.05
        PREP_DIST_MAX, PREP_DIST_MIN = 3.0, 1.0
        W_CROUCH, Z_CROUCH_TARGET, CROUCH_SIGMA = 5.0, 0.9, 0.1
        W_SLOW_DOWN, VEL_X_TARGET, VEL_X_SIGMA = 7.0, 0.3, 0.2
        W_SYMMETRY = 5.0
        W_CLEARANCE, CLEARANCE_MARGIN, W_JUMP = 20.0, 0.1, 2.5
        SUCCESS_BONUS = 50.0
        MIN_SUCCESS_HEIGHT = 1.2
        HEALTHY_Z_THRESHOLD, FALLEN_Z_THRESHOLD = 0.8, 0.7
        W_FALLEN_PENALTY = -50.0
        
        # ==================== v8 추가: 진행도 보상 가중치 ====================
        W_PROGRESS = 40.0
        # ====================================================================

        # --- 관측값 분해 ---
        walker_x, z_torso = obs[0], obs[1]
        thigh_angle, leg_angle = obs[3], obs[4]
        thigh_left_angle, leg_left_angle = obs[6], obs[7]
        vel_x, vel_z, angvel_torso = obs[9], obs[10], obs[11]
        bump1_dx, bump1_h, bump1_w = custom_obs[17], custom_obs[18], custom_obs[19]

        # 1. 기반 보상 (기존과 동일)
        alive_bonus = W_ALIVE if z_torso > HEALTHY_Z_THRESHOLD else 0.0
        base_reward = (W_FORWARD * vel_x) + alive_bonus + \
                      (W_CTRL * np.sum(np.square(action))) + \
                      (W_STABILITY * np.square(angvel_torso))

        # 2. 적응형 준비 보상 (기존과 동일)
        shaping_reward = 0.0
        if bump1_dx >= 0:
            prep_intensity = np.clip((bump1_dx - PREP_DIST_MIN) / (PREP_DIST_MAX - PREP_DIST_MIN), 0, 1)
            is_approaching = (self.prev_bump1_dx > bump1_dx)

            if prep_intensity > 0 and is_approaching:
                # ... (기존 준비 보상 로직)
                preparation_scale = min(1.0, bump1_h / 0.5)
                slowing_down_reward = W_SLOW_DOWN * np.exp(-((vel_x - VEL_X_TARGET)**2) / (2 * VEL_X_SIGMA**2))
                symmetry_reward = W_SYMMETRY * np.exp(-((thigh_angle - thigh_left_angle)**2 + (leg_angle - leg_left_angle)**2))
                crouch_reward = W_CROUCH * np.exp(-((z_torso - Z_CROUCH_TARGET)**2) / (2 * CROUCH_SIGMA**2))
                preparation_reward = (slowing_down_reward + symmetry_reward + crouch_reward) * preparation_scale
                shaping_reward += preparation_reward * prep_intensity
            
            if 0 <= bump1_dx <= PREP_DIST_MIN:
                clearance_reward = W_CLEARANCE * max(0, z_torso - (bump1_h * 2 + CLEARANCE_MARGIN))
                jump_reward = W_JUMP * vel_z
                shaping_reward += clearance_reward + jump_reward
        
        # ==================== 3. v8: 진행도(Progress) 기반 조밀한 보상 ====================
        clearing_progress_reward = 0.0
        current_bump_name = f"bump{self.cleared_bumps_count + 1}"
        is_on_top = self._is_foot_on_bump(current_bump_name)

        if is_on_top and bump1_dx >= 0:
            # 장애물의 시작점과 끝점 계산 (x좌표 기준)
            bump_center_x = walker_x + bump1_dx
            bump_start_x = bump_center_x - bump1_w
            bump_end_x = bump_center_x + bump1_w

            # 장애물 위에서 현재 진행도 계산 (0.0 ~ 1.0)
            current_progress = (walker_x - bump_start_x) / (bump_end_x - bump_start_x)
            current_progress = np.clip(current_progress, 0.0, 1.0)

            # 이전 스텝보다 '전진'했을 때만 보상
            delta_progress = current_progress - self.last_progress_on_bump
            if delta_progress > 0:
                clearing_progress_reward = W_PROGRESS * delta_progress

            # 현재 진행도를 다음 스텝을 위해 저장
            self.last_progress_on_bump = current_progress
        else:
            # 장애물 위에 있지 않으면 진행도 리셋
            self.last_progress_on_bump = 0.0
        # =================================================================================

        # 4. 최종 성공 보너스 (기존과 동일)
        success_bonus = 0.0
        next_bump_info = self._get_next_n_bumps_info(n=1)
        next_bump_x_pos = next_bump_info[0][3] if next_bump_info else -1.0
        
        if self.current_bump_target_x > 0 and self.current_bump_target_x != next_bump_x_pos and z_torso > MIN_SUCCESS_HEIGHT:
            success_bonus = SUCCESS_BONUS
            self.cleared_bumps_count += 1
        self.current_bump_target_x = next_bump_x_pos

        # 5. 넘어짐 페널티 (기존과 동일)
        fallen_penalty = 0.0
        if z_torso < FALLEN_Z_THRESHOLD:
            fallen_penalty = W_FALLEN_PENALTY

        self.prev_bump1_dx = bump1_dx

        # --- 최종 보상 합산 ---
        total_reward = base_reward + shaping_reward + clearing_progress_reward + success_bonus + fallen_penalty
        return total_reward, success_bonus
