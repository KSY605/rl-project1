import numpy as np
import gymnasium as gym
import os

class CustomEnvWrapper(gym.Wrapper):
    """
    Walker2d 환경에 bump 장애물 통과를 위한 커스텀 관측 및 보상 로직을 추가하는 래퍼입니다.
    - Observation: [기존 obs(17)] + [dx, h, w] (거리, 높이, 폭)
    - Reward: 단계별 보상 쉐이핑 (준비 -> 점프 -> 성공)
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
            healthy_z_range=(0.5, 10.0),
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

        # --- Observation Space 수정 ---
        # 기존 17 + dx, h, w = 20
        sample_obs, _ = self.reset()
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=sample_obs.shape, dtype=np.float64
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.cleared_bumps_count = 0
        self.current_bump_target_x = self._next_bump_info()[3] # bump_x_pos 인덱스 변경
        return self.custom_observation(obs), info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        
        custom_obs = self.custom_observation(obs)
        custom_reward, success_bonus = self._calculate_rewards(obs, custom_obs, action)
        
        info['cleared_bumps'] = self.cleared_bumps_count
        if success_bonus > 0:
            info['event'] = 'BUMP_CLEARED'
            
        return custom_obs, custom_reward, terminated, truncated, info

    def _next_bump_info(self):
        """
        다음 bump의 정보를 반환합니다. (폭 정보 추가)
        
        Returns:
            (dx, h, w, bump_x_pos):
            - dx: x축 거리
            - h: 절반 높이
            - w: 절반 폭 (size[0])
            - bump_x_pos: 절대 x좌표
        """
        base_env = self.unwrapped
        data, model = base_env.data, base_env.model

        walker_x = data.qpos[0]
        min_dx, height, width, bump_x_pos = np.inf, 0.0, 0.0, np.inf

        for gid in self.bump_geom_ids:
            current_bump_x_pos = data.geom_xpos[gid][0]
            dx = current_bump_x_pos - walker_x
            if dx >= 0.0 and dx < min_dx:
                min_dx = dx
                # size[0]은 x축 방향의 절반 폭, size[2]는 z축 방향의 절반 높이
                width = model.geom_size[gid][0] 
                height = model.geom_size[gid][2]
                bump_x_pos = current_bump_x_pos
        
        if np.isinf(min_dx):
            return -1.0, 0.0, 0.0, -1.0
            
        return min_dx, height, width, bump_x_pos

    def custom_observation(self, obs):
        """관측에 다음 bump의 거리, 높이, 폭 정보를 추가합니다."""
        dx, h, w, _ = self._next_bump_info()
        return np.concatenate([obs, np.array([dx, h, w], dtype=np.float64)])
    
    def _calculate_rewards(self, obs, custom_obs, action):
        # --- 하이퍼파라미터 (통합 전략) ---
        # 계층 1: 기반 보상
        W_FORWARD = 1.2       # 전진 보상의 가치를 약간 조정
        W_ALIVE = 0.1
        W_CTRL = -0.01
        W_STABILITY = -0.05
        
        # 계층 2: 행동 유도 보상 (모든 장애물에 공통 적용)
        PREP_DIST = 2.5
        JUMP_DIST = 0.8
        
        # A. 점프 준비 보상
        W_CROUCH = 5.0
        Z_CROUCH_TARGET = 0.9
        CROUCH_SIGMA = 0.1
        W_SLOW_DOWN = 7.0
        VEL_X_TARGET = 0.2
        VEL_X_SIGMA = 0.2
        W_SYMMETRY = 5.0
        
        # B. 점프 실행 보상
        W_CLEARANCE = 20.0
        CLEARANCE_MARGIN = 0.1
        W_JUMP = 2.5
        
        # C. 파쿠르 보상
        W_PARKOUR = 15.0
        
        # 계층 3: 목표 달성 보상
        SUCCESS_BONUS = 50.0
        MIN_SUCCESS_HEIGHT = 1.2
        # ----------------------------------------------------
        
        # 관측값 분해
        walker_x, z_torso = obs[0], obs[1]
        thigh_angle, leg_angle = obs[3], obs[4]
        thigh_left_angle, leg_left_angle = obs[6], obs[7]
        vel_x, vel_z, angvel_torso = obs[9], obs[10], obs[11]
        dx, h, w = custom_obs[17], custom_obs[18], custom_obs[19]
        
        # 1. 기반 보상 (항상 적용)
        base_reward = (W_FORWARD * vel_x) + W_ALIVE + \
                      (W_CTRL * np.sum(np.square(action))) + \
                      (W_STABILITY * np.square(angvel_torso))

        # 2. 행동 유도 보상 (조건 분기 없이 통합)
        shaping_reward = 0.0
        if dx >= 0: # 장애물이 앞에 있을 때
            # 점프 준비 구간 (PREP_DIST ~ JUMP_DIST)
            if JUMP_DIST < dx <= PREP_DIST:
                # 감속, 대칭, 웅크리기 보상을 항상 계산
                slowing_down_reward = W_SLOW_DOWN * np.exp(-((vel_x - VEL_X_TARGET)**2) / (2 * VEL_X_SIGMA**2))
                thigh_diff = thigh_angle - thigh_left_angle
                leg_diff = leg_angle - leg_left_angle
                symmetry_reward = W_SYMMETRY * np.exp(-(thigh_diff**2 + leg_diff**2))
                crouch_reward = W_CROUCH * np.exp(-((z_torso - Z_CROUCH_TARGET)**2) / (2 * CROUCH_SIGMA**2))
                
                shaping_reward += slowing_down_reward + symmetry_reward + crouch_reward
            
            # 점프 실행 구간 (JUMP_DIST ~ 0)
            elif 0 <= dx <= JUMP_DIST:
                clearance_reward = W_CLEARANCE * max(0, z_torso - (h * 2 + CLEARANCE_MARGIN))
                jump_reward = W_JUMP * vel_z
                shaping_reward += clearance_reward + jump_reward

        # 3. 파쿠르 보상 (장애물 바로 위)
        parkour_reward = 0.0
        if (dx < w) and (dx > -w):
            is_on_top = (h * 2 < z_torso < h * 2 + 0.4) and abs(vel_z) < 0.5
            if is_on_top:
                bump_center_x = walker_x + dx
                dist_to_center = abs(walker_x - bump_center_x)
                parkour_reward = W_PARKOUR * np.exp(-(dist_to_center**2 / (w**2)))
        
        # 4. 최종 성공 보너스
        success_bonus = 0.0
        _, _, _, next_bump_x = self._next_bump_info()
        
        if self.current_bump_target_x > 0 and self.current_bump_target_x != next_bump_x and z_torso > MIN_SUCCESS_HEIGHT:
            success_bonus = SUCCESS_BONUS
            self.cleared_bumps_count += 1
            
        self.current_bump_target_x = next_bump_x

        total_reward = base_reward + shaping_reward + parkour_reward + success_bonus
        return total_reward, success_bonus




    ###### 현재까지 Best #######
    # def _calculate_rewards(self, obs, custom_obs, action):
    #     # 하이퍼파라미터 (튜닝 필요)
    #     W_FORWARD = 1.5
    #     W_ALIVE = 0.1
    #     W_CTRL = -0.01
    #     W_STABILITY = -0.05
    #     PREP_DIST = 2.5
    #     JUMP_DIST = 0.8
    #     W_CROUCH = 4.0
    #     Z_CROUCH_TARGET = 0.9
    #     CROUCH_SIGMA = 0.1
    #     W_CLEARANCE = 20.0
    #     CLEARANCE_MARGIN = 0.1
    #     W_JUMP = 2.5
    #     SUCCESS_BONUS = 50.0
        
    #     # 관측값 분해 (인덱스 변경에 주의)
    #     z_torso = obs[1]
    #     vel_x = obs[9]
    #     vel_z = obs[10]
    #     angvel_torso = obs[11]
    #     dx, h, w = custom_obs[17], custom_obs[18], custom_obs[19] # obs 20개
        
    #     # 보상 계산 로직은 이전과 동일하게 사용 가능
    #     forward_reward = W_FORWARD * vel_x
    #     alive_bonus = W_ALIVE
    #     control_cost = W_CTRL * np.sum(np.square(action))
    #     stability_penalty = W_STABILITY * np.square(angvel_torso)
    #     base_reward = forward_reward + alive_bonus + control_cost + stability_penalty
        
    #     shaping_reward = 0.0
    #     if dx >= 0:
    #         if JUMP_DIST < dx <= PREP_DIST:
    #             crouch_reward = W_CROUCH * np.exp(-((z_torso - Z_CROUCH_TARGET)**2) / (2 * CROUCH_SIGMA**2))
    #             shaping_reward += crouch_reward
            
    #         elif 0 <= dx <= JUMP_DIST:
    #             clearance_target = h * 2 + CLEARANCE_MARGIN
    #             clearance_reward = W_CLEARANCE * max(0, z_torso - clearance_target)
    #             jump_reward = W_JUMP * vel_z
    #             shaping_reward += clearance_reward + jump_reward
                
    #     success_bonus = 0.0
    #     _, _, _, next_bump_x = self._next_bump_info()
        
    #     if self.current_bump_target_x > 0 and self.current_bump_target_x != next_bump_x:
    #         success_bonus = SUCCESS_BONUS
    #         self.cleared_bumps_count += 1
            
    #     self.current_bump_target_x = next_bump_x

    #     total_reward = base_reward + shaping_reward + success_bonus
    #     return total_reward, success_bonus

