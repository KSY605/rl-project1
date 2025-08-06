import numpy as np
import gymnasium as gym
import os
from typing import List, Tuple

class CustomEnvWrapper(gym.Wrapper):
    """
    Walker2d 환경에 지능적인 장애물 통과를 위한 커스텀 로직을 추가하는 래퍼입니다. (v6)

    - 확장된 관측 (Extended Observation):
      [기존 obs(17)] + [가까운 Bump 4개의 정보(dx, h, w) * 4 (12)] = 총 29개
      
    - 지능형 보상 시스템 (Intelligent Reward System):
      1. 동적 파쿠르 보상 (Dynamic Parkour Reward for Staircase)
      2. 적응형 준비 보상 (Adaptive Preparation Reward)
      3. 보상 해킹 방지 (Anti-Reward Hacking Penalties) - v6에서 강화됨
    """

    def __init__(self, render_mode=None, bump_practice=False, bump_challenge=False):
        repo_root = os.path.dirname(os.path.abspath(__file__))
        asset_dir = os.path.join(repo_root, "asset")

        if bump_challenge:
            xml_file = os.path.join(asset_dir, "custom_walker2d_bumps.xml")
        elif bump_practice:
            practice_xml_path = os.path.join(asset_dir, "custom_walker2d_bumps_practice.xml")
            xml_file = practice_xml_path if os.path.exists(practice_xml_path) else os.path.join(asset_dir, "custom_walker2d_bumps.xml")
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

        base_env = env.unwrapped
        base_model = base_env.model
        self.bump_geom_ids = [
            i for i in range(base_model.ngeom)
            if base_model.geom(i).name and base_model.geom(i).name.startswith("bump")
        ]

        self.cleared_bumps_count = 0
        self.current_bump_target_x = np.inf
        # ==================== 보상 해킹 방지 v6 변경점 ====================
        # 이전 스텝의 장애물까지의 거리를 저장하기 위한 변수
        self.prev_bump1_dx = np.inf
        # ===============================================================

        sample_obs, _ = self.reset()
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=sample_obs.shape, dtype=np.float64
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.cleared_bumps_count = 0
        
        next_bump_info = self._get_next_n_bumps_info(n=1)
        if next_bump_info and next_bump_info[0][3] != -1.0:
            self.current_bump_target_x = next_bump_info[0][3]
            # ==================== 보상 해킹 방지 v6 변경점 ====================
            self.prev_bump1_dx = next_bump_info[0][0] # dx 값으로 초기화
            # ===============================================================
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

    def _get_next_n_bumps_info(self, n: int = 4) -> List[Tuple[float, float, float, float]]:
        base_env = self.unwrapped
        data, model = base_env.data, base_env.model
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
        next_bumps_info = self._get_next_n_bumps_info(n=4)
        obs_bumps_info = [info[:3] for info in next_bumps_info]
        flat_bumps_info = np.array(obs_bumps_info, dtype=np.float64).flatten()
        return np.concatenate([obs, flat_bumps_info])
    
    def _calculate_rewards(self, obs: np.ndarray, custom_obs: np.ndarray, action: np.ndarray) -> Tuple[float, float]:
        # --- 하이퍼파라미터 ---
        W_FORWARD, W_ALIVE, W_CTRL, W_STABILITY = 1.2, 0.1, -0.01, -0.05
        PREP_DIST_MAX, PREP_DIST_MIN = 3.0, 1.0 # 준비 자세를 취하기 시작하는 최대/최소 거리 (기존 JUMP_DIST, PREP_DIST)
        W_CROUCH, Z_CROUCH_TARGET, CROUCH_SIGMA = 5.0, 0.9, 0.1
        W_SLOW_DOWN, VEL_X_TARGET, VEL_X_SIGMA = 7.0, 0.2, 0.2
        W_SYMMETRY = 5.0
        W_CLEARANCE, CLEARANCE_MARGIN, W_JUMP = 20.0, 0.1, 2.5
        
        W_PARKOUR = 15.0
        W_STAIRCASE_PENALTY = -20.0
        CONTINUOUS_BUMP_THRESHOLD = 2.0
        STAIRCASE_HEIGHT_THRESHOLD = 0.2
        MAX_PREPARATION_HEIGHT = 0.5

        SUCCESS_BONUS = 50.0
        MIN_SUCCESS_HEIGHT = 1.2
        HEALTHY_Z_THRESHOLD, FALLEN_Z_THRESHOLD = 0.8, 0.7
        W_FALLEN_PENALTY = -50.0
        
        # --- 관측값 분해 ---
        walker_x, z_torso = obs[0], obs[1]
        thigh_angle, leg_angle = obs[3], obs[4]
        thigh_left_angle, leg_left_angle = obs[6], obs[7]
        vel_x, vel_z, angvel_torso = obs[9], obs[10], obs[11]
        
        bump1_dx, bump1_h, bump1_w = custom_obs[17], custom_obs[18], custom_obs[19]
        bump2_dx, bump2_h, _ = custom_obs[20], custom_obs[21], custom_obs[22]

        # --- 보상 계산 ---
        # 1. 기반 보상
        alive_bonus = W_ALIVE if z_torso > HEALTHY_Z_THRESHOLD else 0.0
        base_reward = (W_FORWARD * vel_x) + alive_bonus + \
                      (W_CTRL * np.sum(np.square(action))) + \
                      (W_STABILITY * np.square(angvel_torso))

        # 2. 적응형 준비 보상 (보상 해킹 방지 로직 적용)
        shaping_reward = 0.0
        if bump1_dx >= 0:
            # ==================== 보상 해킹 방지 v6 변경점 ====================
            # 2-1. 보상 연속성 부여: 거리에 따라 준비 자세 보상을 점진적으로 조절
            # PREP_DIST_MAX에서 보상이 100% 적용되고, PREP_DIST_MIN에 가까워질수록 0으로 감소합니다.
            # 이를 통해 특정 구간에 머무르려는 행동(해킹)을 방지합니다.
            prep_intensity = np.clip((bump1_dx - PREP_DIST_MIN) / (PREP_DIST_MAX - PREP_DIST_MIN), 0, 1)

            # 2-2. 전진 조건 추가: 장애물을 향해 다가갈 때만 준비 자세 보상을 제공합니다.
            # 뒤로 가거나 제자리에 머무르면서 보상을 타내는 것을 막습니다.
            is_approaching = (self.prev_bump1_dx > bump1_dx)

            if prep_intensity > 0 and is_approaching:
                # 장애물 높이에 따라 준비 자세의 중요도를 조절
                preparation_scale = min(1.0, bump1_h / (MAX_PREPARATION_HEIGHT + 1e-8))
                
                # 준비 행동들 (속도 줄이기, 자세 낮추기, 다리 모으기)
                slowing_down_reward = W_SLOW_DOWN * np.exp(-((vel_x - VEL_X_TARGET)**2) / (2 * VEL_X_SIGMA**2))
                thigh_diff = thigh_angle - thigh_left_angle
                leg_diff = leg_angle - leg_left_angle
                symmetry_reward = W_SYMMETRY * np.exp(-(thigh_diff**2 + leg_diff**2))
                crouch_reward = W_CROUCH * np.exp(-((z_torso - Z_CROUCH_TARGET)**2) / (2 * CROUCH_SIGMA**2))
                
                # 최종 준비 보상 = (개별 준비 행동 보상 합) * (장애물 높이 가중치) * (거리에 따른 강도)
                preparation_reward = (slowing_down_reward + symmetry_reward + crouch_reward) * preparation_scale
                shaping_reward += preparation_reward * prep_intensity
            # ===============================================================

            # 점프 구간 보상 (장애물 바로 앞)
            if 0 <= bump1_dx <= PREP_DIST_MIN:
                clearance_reward = W_CLEARANCE * max(0, z_torso - (bump1_h * 2 + CLEARANCE_MARGIN))
                jump_reward = W_JUMP * vel_z
                shaping_reward += clearance_reward + jump_reward

        # 3. 동적 파쿠르 보상 (연속 장애물)
        parkour_reward = 0.0
        is_continuous = (bump2_dx > 0 and (bump2_dx - bump1_dx) < CONTINUOUS_BUMP_THRESHOLD)
        if is_continuous and (-bump1_w < bump1_dx < bump1_w):
            is_on_top = (bump1_h * 2 < z_torso < bump1_h * 2 + 0.5) and abs(vel_z) < 0.5
            if is_on_top:
                height_diff = bump2_h - bump1_h
                if height_diff > STAIRCASE_HEIGHT_THRESHOLD:
                    parkour_reward = W_STAIRCASE_PENALTY
                else:
                    bump_center_x = walker_x + bump1_dx
                    forward_edge_x = bump_center_x + bump1_w * 0.8
                    dist_to_edge = abs(walker_x - forward_edge_x)
                    parkour_reward = W_PARKOUR * np.exp(-(dist_to_edge**2 / (bump1_w**2 + 1e-8)))
        
        # 4. 최종 성공 보너스
        success_bonus = 0.0
        next_bump_info = self._get_next_n_bumps_info(n=1)
        next_bump_x_pos = next_bump_info[0][3] if next_bump_info else -1.0
        
        if self.current_bump_target_x > 0 and self.current_bump_target_x != next_bump_x_pos and z_torso > MIN_SUCCESS_HEIGHT:
            success_bonus = SUCCESS_BONUS
            self.cleared_bumps_count += 1
        self.current_bump_target_x = next_bump_x_pos

        # 5. 넘어짐 페널티
        fallen_penalty = 0.0
        if z_torso < FALLEN_Z_THRESHOLD:
            fallen_penalty = W_FALLEN_PENALTY

        # ==================== 보상 해킹 방지 v6 변경점 ====================
        # 현재 장애물 거리를 다음 스텝을 위해 저장
        self.prev_bump1_dx = bump1_dx
        # ===============================================================

        total_reward = base_reward + shaping_reward + parkour_reward + success_bonus + fallen_penalty
        return total_reward, success_bonus
