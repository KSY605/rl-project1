import numpy as np
import gymnasium as gym
import os
from typing import List, Tuple, Dict, Optional

class CustomEnvWrapper(gym.Wrapper):
    """
    Walker2d v3 환경을 Optuna와 연동하기 위해 수정한 래퍼입니다.
    - __init__에서 보상 가중치(reward_weights)를 인자로 받아 동적으로 적용합니다.
    """

    def __init__(self, 
                 render_mode: Optional[str] = None, 
                 bump_practice: bool = False, 
                 bump_challenge: bool = False, 
                 reward_weights: Optional[Dict[str, float]] = None):
        
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

        # Optuna로부터 받은 보상 가중치를 적용합니다.
        self.reward_weights = {
            "W_FORWARD": 1.2, "W_ALIVE": 0.1, "W_CTRL": -0.01, "W_STABILITY": -0.05,
            "W_CROUCH": 5.0, "W_SLOW_DOWN": 7.0, "W_SYMMETRY": 5.0,
            "W_CLEARANCE": 20.0, "W_JUMP": 2.5, "W_PARKOUR": 15.0,
            "SUCCESS_BONUS": 50.0, "W_FALLEN_PENALTY": -50.0,
        }
        if reward_weights:
            self.reward_weights.update(reward_weights)

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
        # 클래스 변수에 저장된 보상 가중치를 불러옵니다.
        W = self.reward_weights
        
        # 기타 하이퍼파라미터
        PREP_DIST, JUMP_DIST = 2.5, 0.8
        Z_CROUCH_TARGET, CROUCH_SIGMA = 0.9, 0.1
        VEL_X_TARGET, VEL_X_SIGMA = 0.2, 0.2
        CLEARANCE_MARGIN = 0.1
        CONTINUOUS_BUMP_THRESHOLD = 2.0
        MIN_SUCCESS_HEIGHT = 1.2
        HEALTHY_Z_THRESHOLD, FALLEN_Z_THRESHOLD = 0.8, 0.7
        
        # 관측값 분해
        walker_x, z_torso = obs[0], obs[1]
        thigh_angle, leg_angle = obs[3], obs[4]
        thigh_left_angle, leg_left_angle = obs[6], obs[7]
        vel_x, vel_z, angvel_torso = obs[9], obs[10], obs[11]
        bump1_dx, bump1_h, bump1_w = custom_obs[17], custom_obs[18], custom_obs[19]
        bump2_dx = custom_obs[20]

        # 보상 계산 (W 딕셔너리 사용)
        alive_bonus = W["W_ALIVE"] if z_torso > HEALTHY_Z_THRESHOLD else 0.0
        base_reward = (W["W_FORWARD"] * vel_x) + alive_bonus + \
                      (W["W_CTRL"] * np.sum(np.square(action))) + \
                      (W["W_STABILITY"] * np.square(angvel_torso))

        shaping_reward = 0.0
        if bump1_dx >= 0:
            if JUMP_DIST < bump1_dx <= PREP_DIST:
                slowing_down = W["W_SLOW_DOWN"] * np.exp(-((vel_x - VEL_X_TARGET)**2) / (2 * VEL_X_SIGMA**2))
                symmetry = W["W_SYMMETRY"] * np.exp(-((thigh_angle - thigh_left_angle)**2 + (leg_angle - leg_left_angle)**2))
                crouch = W["W_CROUCH"] * np.exp(-((z_torso - Z_CROUCH_TARGET)**2) / (2 * CROUCH_SIGMA**2))
                shaping_reward += slowing_down + symmetry + crouch
            elif 0 <= bump1_dx <= JUMP_DIST:
                clearance = W["W_CLEARANCE"] * max(0, z_torso - (bump1_h * 2 + CLEARANCE_MARGIN))
                jump = W["W_JUMP"] * vel_z
                shaping_reward += clearance + jump

        parkour_reward = 0.0
        is_continuous = (bump2_dx > 0 and (bump2_dx - bump1_dx) < CONTINUOUS_BUMP_THRESHOLD)
        if is_continuous and (-bump1_w < bump1_dx < bump1_w):
            is_on_top = (bump1_h * 2 < z_torso < bump1_h * 2 + 0.4) and abs(vel_z) < 0.5
            if is_on_top:
                dist_to_center = abs(walker_x - (walker_x + bump1_dx))
                parkour_reward = W["W_PARKOUR"] * np.exp(-(dist_to_center**2 / (bump1_w**2 + 1e-8)))
        
        success_bonus = 0.0
        next_bump_x_pos = self._get_next_n_bumps_info(n=1)[0][3]
        if self.current_bump_target_x > 0 and self.current_bump_target_x != next_bump_x_pos and z_torso > MIN_SUCCESS_HEIGHT:
            success_bonus = W["SUCCESS_BONUS"]
            self.cleared_bumps_count += 1
        self.current_bump_target_x = next_bump_x_pos

        fallen_penalty = W["W_FALLEN_PENALTY"] if z_torso < FALLEN_Z_THRESHOLD else 0.0

        total_reward = base_reward + shaping_reward + parkour_reward + success_bonus + fallen_penalty
        return total_reward, success_bonus
