import numpy as np
import gymnasium as gym
import os
from typing import List, Tuple, Set

class CustomEnvWrapper(gym.Wrapper):
    """
    Walker2d 환경에 지능적인 장애물 통과를 위한 커스텀 로직을 추가하는 래퍼입니다. (XML 기반 커리큘럼)
    """

    def __init__(self, render_mode=None, bump_practice=False, bump_challenge=False, curriculum_level: int = 0):
        repo_root = os.path.dirname(os.path.abspath(__file__))
        asset_dir = os.path.join(repo_root, "asset")

        # ==================== CURRICULUM: 변경점 ====================
        # curriculum_level에 따라 사용할 XML 파일을 동적으로 선택합니다.
        # bump_challenge 플래그는 최종 테스트에만 사용하도록 합니다.
        xml_file = None
        if curriculum_level > 0:
            # 커리큘럼 학습 중일 경우, 해당 레벨의 XML 파일을 로드합니다.
            xml_file_path = os.path.join(asset_dir, f"curriculum_level{curriculum_level}.xml")
            if os.path.exists(xml_file_path):
                xml_file = xml_file_path
                print(f"--- Loading Curriculum Level {curriculum_level}: {os.path.basename(xml_file)} ---")
            else:
                print(f"--- WARNING: curriculum_level{curriculum_level}.xml not found. Using default environment. ---")
        elif bump_challenge:
            # 최종 챌린지 환경
            xml_file = os.path.join(asset_dir, "custom_walker2d_bumps_v2.xml")
            print(f"--- Loading Final Challenge: custom_walker2d_bumps_v2.xml ---")
        elif bump_practice:
            # 연습 환경
            xml_file = os.path.join(asset_dir, "custom_walker2d_bumps_practice_v2.xml")
        # else: 기본 Walker2D (장애물 없음)
        # =============================================================

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
        
        # bump_geom_ids를 찾는 로직은 그대로 유지합니다.
        # 이제 이 로직은 각 XML 파일에 정의된 bump들만 찾게 됩니다.
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
        self.last_progress_on_bump = 0.0

        sample_obs, _ = self.reset()
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=sample_obs.shape, dtype=np.float64
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.cleared_bumps_count = 0
        self.last_progress_on_bump = 0.0
        
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
        data, model = self.base_env.data, self.base_env.model
        walker_x = data.qpos[0]
        
        upcoming_bumps = []
        # 이 부분은 수정할 필요가 없습니다. self.bump_geom_ids에 이미 현재 레벨의 장애물만 들어있습니다.
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
        # 보상 로직은 변경할 필요 없습니다.
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
        W_PROGRESS = 40.0

        walker_x, z_torso = obs[0], obs[1]
        thigh_angle, leg_angle = obs[3], obs[4]
        thigh_left_angle, leg_left_angle = obs[6], obs[7]
        vel_x, vel_z, angvel_torso = obs[9], obs[10], obs[11]
        bump1_dx, bump1_h, bump1_w = custom_obs[17], custom_obs[18], custom_obs[19]

        alive_bonus = W_ALIVE if z_torso > HEALTHY_Z_THRESHOLD else 0.0
        base_reward = (W_FORWARD * vel_x) + alive_bonus + (W_CTRL * np.sum(np.square(action))) + (W_STABILITY * np.square(angvel_torso))

        shaping_reward = 0.0
        if bump1_dx >= 0:
            prep_intensity = np.clip((bump1_dx - PREP_DIST_MIN) / (PREP_DIST_MAX - PREP_DIST_MIN), 0, 1)
            is_approaching = (self.prev_bump1_dx > bump1_dx)
            if prep_intensity > 0 and is_approaching:
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
        
        clearing_progress_reward = 0.0
        current_bump_name = f"bump{self.cleared_bumps_count + 1}"
        is_on_top = self._is_foot_on_bump(current_bump_name)
        if is_on_top and bump1_dx >= 0:
            bump_center_x = walker_x + bump1_dx
            bump_start_x = bump_center_x - bump1_w
            bump_end_x = bump_center_x + bump1_w
            current_progress = np.clip((walker_x - bump_start_x) / (bump_end_x - bump_start_x), 0.0, 1.0)
            delta_progress = current_progress - self.last_progress_on_bump
            if delta_progress > 0:
                clearing_progress_reward = W_PROGRESS * delta_progress
            self.last_progress_on_bump = current_progress
        else:
            self.last_progress_on_bump = 0.0

        success_bonus = 0.0
        next_bump_info = self._get_next_n_bumps_info(n=1)
        next_bump_x_pos = next_bump_info[0][3] if next_bump_info else -1.0
        if self.current_bump_target_x > 0 and self.current_bump_target_x != next_bump_x_pos and z_torso > MIN_SUCCESS_HEIGHT:
            success_bonus = SUCCESS_BONUS
            self.cleared_bumps_count += 1
        self.current_bump_target_x = next_bump_x_pos

        fallen_penalty = 0.0
        if z_torso < FALLEN_Z_THRESHOLD:
            fallen_penalty = W_FALLEN_PENALTY

        self.prev_bump1_dx = bump1_dx
        total_reward = base_reward + shaping_reward + clearing_progress_reward + success_bonus + fallen_penalty
        return total_reward, success_bonus
