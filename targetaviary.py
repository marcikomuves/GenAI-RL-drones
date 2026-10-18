from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType
import pybullet as p
from gymnasium import spaces
import numpy as np
import sys
from collections import deque
from targetstategenerator import TargetStateGenerator
from dronestategenerator import DroneStateGenerator

class TargetAviary(BaseAviary):
    def __init__(
        self,
        target_state_generator: TargetStateGenerator,
        init_state_generator: DroneStateGenerator,
        max_t = None,
        max_dist = 3.0,
        max_ang_vel = 4.0 * np.pi,
        act: ActionType = ActionType.RPM,
        drone_model: DroneModel = DroneModel.CF2X,
        neighbourhood_radius: float = np.inf,
        physics: Physics = Physics.PYB,
        pyb_freq: int = 240,
        ctrl_freq: int = 240,
        gui=False,
        record=False        
    ):
        if act is not ActionType.RPM:
            print(f"[ERROR] Action type {act} is not supported.")
            sys.exit(1)

        self.ACT_TYPE = act
        self.target_state_generator = target_state_generator
        self.init_state_gen = init_state_generator
        self.lookahead_dt = 1.0/ctrl_freq
        self.t = 0
        self.MAX_T = max_t
        self.DIST_LIMIT_M = max_dist
        self.ANG_VEL_LIMIT = max_ang_vel
        self.MAX_SPEED_MPS = 0
        self.SPEED_LIMIT_MPS = 0
        self.current_engine_cmd = np.zeros(4)
        
        super().__init__(
            drone_model=drone_model,
            num_drones=1,
            neighbourhood_radius=neighbourhood_radius,
            physics=physics,
            pyb_freq=pyb_freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=record
        )

        self.MAX_SPEED_MPS = self.MAX_SPEED_KMH * (1000/3600)
        self.SPEED_LIMIT_MPS = 0.3 * self.MAX_SPEED_MPS
        self.observation_space = self._observationSpace()
       

    def _actionSpace(self):
        return spaces.Box(
            low=np.repeat(-1, 4), 
            high=np.repeat(1, 4), 
            dtype=np.float32
        )
    

    def _observationLimits(self):
        return spaces.Box(
            high=np.concatenate(
                [
                    np.repeat(self.DIST_LIMIT_M, 3),        # e_p_body's components cannot be larger than MAX_DIST = 3 (termination condition)
                    np.repeat(self.SPEED_LIMIT_MPS, 3),     # vel_body's components cannot be larger than SPEED_LIMIT_MPS (termination condition)
                    np.repeat(1.0, 3),                      # gravity_body's components cannot be larger than 1 (normalized)
                    np.repeat(self.ANG_VEL_LIMIT, 3),       # ang_vel_body's components cannot be larger than ANG_VEL_LIMIT = PI (termination condition)
                    np.repeat(self._actionSpace().high, 1)  # action space bounds
                ]
            ), 
            low=np.concatenate(
                [
                    np.repeat(-self.DIST_LIMIT_M, 3),
                    np.repeat(-self.SPEED_LIMIT_MPS, 3), 
                    np.repeat(-1.0, 3), 
                    np.repeat(-self.ANG_VEL_LIMIT, 3), 
                    np.repeat(self._actionSpace().low, 1)
                ]
            ), 
            dtype=np.float32
        )

    def _observationSpace(self):
        return spaces.Box(-1, 1, (16,))
        
    
    def _computeObs(self):
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        quat = state[3:7]
        R = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        vel = state[10:13]
        ang_vel = state[13:16]
        p_ref, v_ref = self.target_state_generator.get(self.t)
        e_p_world = p_ref - pos
        e_p_body = R.T @ e_p_world
        e_v_world = v_ref - vel
        e_v_body = R.T @ e_v_world
        gravity_body = R.T @ np.array([0, 0, -1.0])
        ang_vel_body = R.T @ ang_vel
        
        obs = np.concatenate(
            [
                e_p_body,
                e_v_body,
                gravity_body,
                ang_vel_body,
                self.current_engine_cmd
            ], 
            axis=None
        )

        assert obs.shape == self.observation_space.shape
        normalized_obs = obs/self._observationLimits().high
        return np.clip(normalized_obs, -1.0, 1.0)

    def _computeReward(self):
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        quat = state[3:7]
        vel = state[10:13]
        ang_vel = state[13:16]
        R = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3) # columns of R = body-frame unit axes expressed in world coordinates
        p_ref, v_ref = self.target_state_generator.get(self.t)
        e_p_world = p_ref - pos
        dist = np.linalg.norm(e_p_world)
        e_v_world = v_ref - vel

        # 1) reward proximity to target
        r_prox = 5.0 * np.exp(-3.0 * dist)
        # 2) velocity error
        #r_vel = 2.0 * np.exp(-2.0 * np.linalg.norm(e_p_world + e_v_world * self.lookahead_dt))
        if hasattr(self, "prev_dist"): 
            r_progress = 2.0 * np.tanh(5.0 * (self.prev_dist - dist))
        else:
            r_progress = 0.0
        self.prev_dist = dist
        # 3) reward upright orientation
        #r_upright = R[2, 2]
        # 4) penalize large angular velocities
        r_ang_vel = -0.1 * np.linalg.norm(ang_vel)
        

        return (
            + r_prox  
            + r_progress
            #+ r_upright
            + r_ang_vel
        )
    

    def _computeTerminated(self):
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        rpy = state[7:10]
        vel = state[10:13]
        ang_vel = state[13:16]
        p_ref, _ = self.target_state_generator.get(self.t)

        # Too far from target
        if np.linalg.norm(pos - p_ref) > self.DIST_LIMIT_M:
            return True

        # Too large velocity
        #if np.linalg.norm(vel) > self.SPEED_LIMIT_MPS:
        #    return True

        # Too large angular velocity
        #if np.linalg.norm(ang_vel) > self.ANG_VEL_LIMIT:
        #    return True
        
        # Flipped
        if abs(rpy[0]) > np.pi/2 or abs(rpy[1]) > np.pi/2:
            return True
        
        # On the ground
        if pos[2] <= 0.014:
            return True
        
        return False
    

    def _computeTruncated(self):
        if self.MAX_T is not None:
            return self.t >= self.MAX_T
        else:
            return False
    
    def _computeInfo(self):
        return {"answer": 42}
    

    def _preprocessAction(self, action):
        action = action.reshape(-1) # always reshape into (A,) form
        # rpm = np.zeros((1, 4))
        # rpm[0, :] = np.clip(self.HOVER_RPM * (1 + 0.2 * action), 0, self.MAX_RPM)
        rpm = super()._normalizedActionToRPM(action)
        return rpm
    

    def step(self, action):
        self.current_engine_cmd = action
        self.t += self.lookahead_dt
        _, reward, terminated, truncated, info = super().step(action) # calls self._computeReward AFTER the simulation step
        return self._computeObs(), reward, terminated, truncated, info
    

    def reset(self, seed=None, options=None):
        _, info = super().reset(seed=seed, options=options)
        self.init_state_gen.reset(self.np_random)
        self.target_state_generator.reset(self.np_random)
        self.t = 0
        self.current_engine_cmd = np.zeros(4)
        xyz, vel, rpy, ang_vel = self.init_state_gen.get()
        drone_id = self.getDroneIds()[0]
        p.resetBasePositionAndOrientation(
            drone_id, 
            posObj=xyz, 
            ornObj=p.getQuaternionFromEuler(rpy.tolist()),
            physicsClientId=self.getPyBulletClient()
        )
        p.resetBaseVelocity(
            drone_id,
            linearVelocity=vel.tolist(),
            angularVelocity=ang_vel.tolist(),
            physicsClientId=self.getPyBulletClient()
        )        
        self._updateAndStoreKinematicInformation()
        if hasattr(self, "prev_dist"):
            del self.prev_dist
        return self._computeObs(), info