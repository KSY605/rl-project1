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
      1. 상황 인지 파쿠르 보상 (Context-Aware Parkour Reward)
      2. 보상 해킹 방지 (Anti-Reward Hacking Penalties)
      3. 단계별 행동 유도 보상 (Shaped Rewards for Preparation and Jumping)
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
            # 기본 Walker2d 환경 (XML 파일 없음)
            xml_file = None

        env = gym.make(
            "Walker2d-v5",
            xml_file=xml_file,
            render_mode=render_mode,
            exclude_current_positions_from_observation=False,
            frame_skip=10,
            # healthy_z_range를 약간 낮춰서, 넘어지는 것을 환경이 바로 종료시키지 않고
            # 보상 함수가 처리하도록 유도합니다.
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
        # 성공 보너스 계산을 위해, 가장 가까운 bump의 절대 위치를 추적합니다.
        self.current_bump_target_x = np.inf

        # --- Observation Space 수정 ---
        # 기존 17 + (dx, h, w) * 4 = 17 + 12 = 29
        sample_obs, _ = self.reset()
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=sample_obs.shape, dtype=np.float64
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.cleared_bumps_count = 0
        # 리셋 시, 첫 번째 bump의 x좌표를 목표로 설정합니다.
        self.current_bump_target_x = self._get_next_n_bumps_info(n=1)[0][3]
        return self.custom_observation(obs), info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        
        custom_obs = self.custom_observation(obs)
        custom_reward, success_bonus = self._calculate_rewards(obs, custom_obs, action)
        
        # 넘어짐 페널티에 의해 에피소드를 조기 종료시킬 수 있습니다.
        if custom_reward < -40: # W_FALLEN_PENALTY 와 연동
            terminated = True

        info['cleared_bumps'] = self.cleared_bumps_count
        if success_bonus > 0:
            info['event'] = 'BUMP_CLEARED'
            
        return custom_obs, custom_reward, terminated, truncated, info

    def _get_next_n_bumps_info(self, n: int = 4) -> List[Tuple[float, float, float, float]]:
        """
        에이전트 앞에 있는 다음 n개의 bump 정보를 거리순으로 반환합니다.

        Args:
            n (int): 가져올 bump의 개수.

        Returns:
            List[Tuple[float, float, float, float]]: 
            [(dx, h, w, x_pos), ...] 형태의 리스트.
            n개보다 적은 bump가 남으면 [-1.0, 0.0, 0.0, -1.0]으로 채워집니다.
        """
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
        """관측에 다음 4개 bump의 (거리, 높이, 폭) 정보를 순서대로 추가합니다."""
        next_bumps_info = self._get_next_n_bumps_info(n=4)
        
        # x_pos 정보는 관측에 포함하지 않고 dx, h, w 정보만 사용합니다.
        obs_bumps_info = [info[:3] for info in next_bumps_info]
        
        flat_bumps_info = np.array(obs_bumps_info, dtype=np.float64).flatten()
        
        return np.concatenate([obs, flat_bumps_info])
    
    def _calculate_rewards(self, obs: np.ndarray, custom_obs: np.ndarray, action: np.ndarray) -> Tuple[float, float]:
        # --- 하이퍼파라미터 ---
        # 계층 1: 기반 보상
        W_FORWARD = 1.2
        W_ALIVE = 0.1
        W_CTRL = -0.01
        W_STABILITY = -0.05
        
        # 계층 2: 행동 유도 보상
        PREP_DIST = 2.5
        JUMP_DIST = 0.8
        W_CROUCH = 5.0
        Z_CROUCH_TARGET = 0.9
        CROUCH_SIGMA = 0.1
        W_SLOW_DOWN = 7.0
        VEL_X_TARGET = 0.2
        VEL_X_SIGMA = 0.2
        W_SYMMETRY = 5.0
        W_CLEARANCE = 20.0
        CLEARANCE_MARGIN = 0.1
        W_JUMP = 2.5
        
        # 계층 3: 상황 인지 파쿠르 보상
        W_PARKOUR = 15.0
        CONTINUOUS_BUMP_THRESHOLD = 2.0 # 다음 bump가 이 거리 안에 있으면 '연속'으로 간주

        # 계층 4: 목표 달성 및 페널티
        SUCCESS_BONUS = 50.0
        MIN_SUCCESS_HEIGHT = 1.2
        HEALTHY_Z_THRESHOLD = 0.8  # 이 높이 이상이어야 '건강한' 상태
        FALLEN_Z_THRESHOLD = 0.7   # 이 높이 미만이면 '넘어진' 상태
        W_FALLEN_PENALTY = -50.0   # 넘어졌을 때 받을 강력한 페널티
        # ----------------------------------------------------
        
        # --- 관측값 분해 ---
        # 기본 관측
        walker_x, z_torso = obs[0], obs[1]
        thigh_angle, leg_angle = obs[3], obs[4]
        thigh_left_angle, leg_left_angle = obs[6], obs[7]
        vel_x, vel_z, angvel_torso = obs[9], obs[10], obs[11]
        
        # 확장된 Bump 관측 (가장 가까운 bump 2개 정보만 우선 사용)
        bump1_dx, bump1_h, bump1_w = custom_obs[17], custom_obs[18], custom_obs[19]
        bump2_dx = custom_obs[20] # 두 번째 bump까지의 거리

        # --- 보상 계산 ---
        # 1. 기반 보상 (보상 해킹 방지 적용)
        alive_bonus = W_ALIVE if z_torso > HEALTHY_Z_THRESHOLD else 0.0
        base_reward = (W_FORWARD * vel_x) + alive_bonus + \
                      (W_CTRL * np.sum(np.square(action))) + \
                      (W_STABILITY * np.square(angvel_torso))

        # 2. 행동 유도 보상 (가장 가까운 bump 기준)
        shaping_reward = 0.0
        if bump1_dx >= 0:
            if JUMP_DIST < bump1_dx <= PREP_DIST:
                slowing_down_reward = W_SLOW_DOWN * np.exp(-((vel_x - VEL_X_TARGET)**2) / (2 * VEL_X_SIGMA**2))
                thigh_diff = thigh_angle - thigh_left_angle
                leg_diff = leg_angle - leg_left_angle
                symmetry_reward = W_SYMMETRY * np.exp(-(thigh_diff**2 + leg_diff**2))
                crouch_reward = W_CROUCH * np.exp(-((z_torso - Z_CROUCH_TARGET)**2) / (2 * CROUCH_SIGMA**2))
                shaping_reward += slowing_down_reward + symmetry_reward + crouch_reward
            
            elif 0 <= bump1_dx <= JUMP_DIST:
                clearance_reward = W_CLEARANCE * max(0, z_torso - (bump1_h * 2 + CLEARANCE_MARGIN))
                jump_reward = W_JUMP * vel_z
                shaping_reward += clearance_reward + jump_reward

        # 3. 상황 인지 파쿠르 보상
        parkour_reward = 0.0
        # 다음 bump가 연속적이라고 판단될 만큼 가까울 때만 파쿠르 보상 활성화
        is_continuous = (bump2_dx > 0 and (bump2_dx - bump1_dx) < CONTINUOUS_BUMP_THRESHOLD)
        if is_continuous:
            if (-bump1_w < bump1_dx < bump1_w): # bump 위에 있을 때
                is_on_top = (bump1_h * 2 < z_torso < bump1_h * 2 + 0.4) and abs(vel_z) < 0.5
                if is_on_top:
                    bump_center_x = walker_x + bump1_dx
                    dist_to_center = abs(walker_x - bump_center_x)
                    parkour_reward = W_PARKOUR * np.exp(-(dist_to_center**2 / (bump1_w**2)))
        
        # 4. 최종 성공 보너스
        success_bonus = 0.0
        # _get_next_n_bumps_info는 정렬된 리스트를 반환하므로, 첫 번째 요소의 x좌표가 다음 목표가 됨
        next_bump_x_pos = self._get_next_n_bumps_info(n=1)[0][3]
        
        # 현재 목표 bump를 지나쳐 다음 bump를 목표로 삼기 시작했고, 충분히 높은 자세를 유지하고 있다면 성공
        if self.current_bump_target_x > 0 and self.current_bump_target_x != next_bump_x_pos and z_torso > MIN_SUCCESS_HEIGHT:
            success_bonus = SUCCESS_BONUS
            self.cleared_bumps_count += 1
            
        self.current_bump_target_x = next_bump_x_pos

        # 5. 넘어짐 페널티
        fallen_penalty = 0.0
        if z_torso < FALLEN_Z_THRESHOLD:
            fallen_penalty = W_FALLEN_PENALTY

        total_reward = base_reward + shaping_reward + parkour_reward + success_bonus + fallen_penalty
        return total_reward, success_bonus
