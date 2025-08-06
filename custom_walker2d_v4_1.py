import numpy as np
import gymnasium as gym
import os
from typing import List, Tuple

class CustomEnvWrapper(gym.Wrapper):
    """
    Walker2d 환경에 지능적인 장애물 통과를 위한 커스텀 로직을 추가하는 래퍼입니다.

    - 확장된 관측 (Extended Observation):
      [기존 obs(17)] + [가까운 Bump 4개의 정보(dx, h, w) * 4 (12)] = 총 29개
      
    - 지능형 보상 시스템 (Intelligent Reward System):
      1. 동적 파쿠르 보상 (Dynamic Parkour Reward for Staircase)
      2. 적응형 준비 보상 (Adaptive Preparation Reward)
      3. 보상 해킹 방지 (Anti-Reward Hacking Penalties)
      4. [NEW] 발 높이 보상 (Foot Clearance Reward) 및 동적 도약 보상 (Dynamic Jump Reward)
      5. [NEW] 머뭇거림 페널티 (Lingering Penalty)
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
            # xml_file=None 이면 기본 Walker2d-v5 환경이 로드됩니다.
            xml_file = os.path.join(asset_dir, "walker2d.xml")

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

        sample_obs, _ = self.reset()
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=sample_obs.shape, dtype=np.float64
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.cleared_bumps_count = 0
        # 에피소드 시작 시, 다음 bump 정보가 있을 경우에만 목표 x좌표를 설정합니다.
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
        PREP_DIST, JUMP_DIST = 2.5, 0.8
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

        # <<<--- 개선안 1 & 3을 위한 하이퍼파라미터 추가 --->>>
        W_FOOT_CLEARANCE = 15.0  # 발 높이 보상 가중치
        W_LINGERING_PENALTY = -15.0 # 머뭇거림 페널티 가중치
        LINGERING_VEL_THRESHOLD = 0.3 # 머뭇거림으로 판단할 속도 임계값
        
        # --- 관측값 및 상태 분해 ---
        walker_x, z_torso = obs[0], obs[1]
        thigh_angle, leg_angle = obs[3], obs[4]
        thigh_left_angle, leg_left_angle = obs[6], obs[7]
        vel_x, vel_z, angvel_torso = obs[9], obs[10], obs[11]
        
        bump1_dx, bump1_h, bump1_w = custom_obs[17], custom_obs[18], custom_obs[19]
        bump2_dx, bump2_h, _ = custom_obs[20], custom_obs[21], custom_obs[22]
        
        # <<<--- 개선안 1을 위해 발 위치 정보 가져오기 --->>>
        foot_z = self.unwrapped.data.body('foot').xpos[2]
        foot_left_z = self.unwrapped.data.body('foot_left').xpos[2]


        # --- 보상 계산 ---
        # 1. 기반 보상
        alive_bonus = W_ALIVE if z_torso > HEALTHY_Z_THRESHOLD else 0.0
        base_reward = (W_FORWARD * vel_x) + alive_bonus + \
                      (W_CTRL * np.sum(np.square(action))) + \
                      (W_STABILITY * np.square(angvel_torso))

        # 2. 적응형 준비 보상
        shaping_reward = 0.0
        if bump1_dx >= 0:
            if JUMP_DIST < bump1_dx <= PREP_DIST:
                preparation_scale = min(1.0, bump1_h / (MAX_PREPARATION_HEIGHT + 1e-8))
                
                slowing_down_reward = W_SLOW_DOWN * np.exp(-((vel_x - VEL_X_TARGET)**2) / (2 * VEL_X_SIGMA**2))
                thigh_diff = thigh_angle - thigh_left_angle
                leg_diff = leg_angle - leg_left_angle
                symmetry_reward = W_SYMMETRY * np.exp(-(thigh_diff**2 + leg_diff**2))
                crouch_reward = W_CROUCH * np.exp(-((z_torso - Z_CROUCH_TARGET)**2) / (2 * CROUCH_SIGMA**2))
                
                shaping_reward += (slowing_down_reward + symmetry_reward + crouch_reward) * preparation_scale
            
            elif 0 <= bump1_dx <= JUMP_DIST:
                # 몸통 높이 보상 (기존)
                clearance_reward = W_CLEARANCE * max(0, z_torso - (bump1_h * 2 + CLEARANCE_MARGIN))
                
                # <<<--- 개선안 2: 동적 도약 보상 --->>>
                # 장애물 높이에 비례하여 도약 보상 가중치를 동적으로 조절
                dynamic_w_jump = W_JUMP * (1.0 + bump1_h * 2.0)
                jump_reward = dynamic_w_jump * vel_z

                # <<<--- 개선안 1: 발 높이 보상 추가 --->>>
                # 양 발 중 더 낮은 발의 높이를 기준으로 보상
                min_foot_z = min(foot_z, foot_left_z)
                foot_clearance_reward = W_FOOT_CLEARANCE * max(0, min_foot_z - (bump1_h * 2 + CLEARANCE_MARGIN))

                shaping_reward += clearance_reward + jump_reward + foot_clearance_reward

        # 3. 동적 파쿠르 보상
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

        # 5. 페널티
        # <<<--- 개선안 3: 머뭇거림 페널티 추가 --->>>
        lingering_penalty = 0.0
        # 에이전트가 첫 번째 장애물 바로 위에 있고 수평 속도가 매우 느릴 경우 페널티 부여
        if abs(bump1_dx) < bump1_w and vel_x < LINGERING_VEL_THRESHOLD:
            lingering_penalty = W_LINGERING_PENALTY
        
        # 넘어짐 페널티
        fallen_penalty = 0.0
        if z_torso < FALLEN_Z_THRESHOLD:
            fallen_penalty = W_FALLEN_PENALTY

        total_reward = (base_reward + 
                        shaping_reward + 
                        parkour_reward + 
                        success_bonus + 
                        fallen_penalty + 
                        lingering_penalty)
                        
        return total_reward, success_bonus
