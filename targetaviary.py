from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType
import pybullet as p
from gymnasium import spaces
import numpy as np
import sys
from targetstategenerator import TargetStateGenerator
from dronestategenerator import DroneStateGenerator

KP = 0.7
KV = 2.8
S_P = 3.0          # max distance
S_V = 4.0          # ≈ p95 of |d_v| (3.6), with a little headroom
S_A = KV * S_V

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
        
        assert act is ActionType.RPM

        self.target_state_generator = target_state_generator
        self.init_state_gen = init_state_generator
        self.dt = 1.0/ctrl_freq
        self.t = 0
        self.max_t = max_t
        self.max_dist = max_dist
        
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
       

    def _actionSpace(self):

        return spaces.Box(-1, 1, (4,))
    

    def _observationSpace(self):

        return spaces.Box(-np.inf, np.inf, (15,))
        
    
    def _computeObs(self):

        assert hasattr(self, "prev_vel")

        p_ref, v_ref, a_ref = self.target_state_generator.get(self.t)
        d_p = p_ref - self.pos[0]
        d_v = v_ref + KP * d_p - self.vel[0]
        d_a = a_ref + KV * d_v - (self.vel[0] - self.prev_vel) / self.dt

        R = np.array(p.getMatrixFromQuaternion(self.quat[0])).reshape(3, 3)

        obs = np.concatenate(
            [
                R.T @ d_p / S_P,
                R.T @ d_v / S_V,
                R.T @ d_a / S_A,
                R.T @ np.array([0, 0, -1.0]), 
                R.T @ self.ang_v[0]
            ]
        )

        assert obs.shape == self.observation_space.shape
        return obs

    def _computeReward(self):

        assert hasattr(self, "prev_dist") and hasattr(self, "prev_act")

        p_ref, v_ref, _ = self.target_state_generator.get(self.t)

        d_p = p_ref - self.pos[0]
        dist = np.linalg.norm(d_p)
        d_v = v_ref + KP * d_p - self.vel[0] # same cascade as the observation

        r_prox = 5.0 * np.exp(-3.0 * dist)
        r_prog = 1.0 * np.clip((self.prev_dist - dist) * self.CTRL_FREQ, -1.0, 1.0) # m/s toward target
        r_vel  = -0.5 * min(np.linalg.norm(d_v), 3.0)      # "be on the approach profile"
        r_act  = -0.05 * np.linalg.norm(self.last_clipped_action / self.MAX_RPM - self.prev_act)
        R = np.array(p.getMatrixFromQuaternion(self.quat[0])).reshape(3, 3)
        w_body = R.T @ self.ang_v[0]
        r_ang = -0.1 * np.linalg.norm(w_body[:2]) - 0.4 * abs(w_body[2])

        return r_prox + r_prog + r_vel + r_act + r_ang
    

    def _computeTerminated(self):
        p_ref, _, _ = self.target_state_generator.get(self.t)

        # Too far from target
        if np.linalg.norm(self.pos[0] - p_ref) > self.max_dist:
            return True
        
        # Flipped
        if abs(self.rpy[0][0]) > np.pi/2 or abs(self.rpy[0][1]) > np.pi/2:
            return True
        
        # On the ground
        if self.pos[0][2] <= 0.014:
            return True
        
        return False
    

    def _computeTruncated(self):
        if self.max_t is not None:
            return self.t >= self.max_t
        else:
            return False

    
    def _computeInfo(self):

        return {"answer": 42}
    

    def _preprocessAction(self, action):
        action = action.reshape(-1)
        # rpm = np.zeros((1, 4))
        # rpm[0, :] = np.clip(self.HOVER_RPM * (1 + 0.2 * action), 0, self.MAX_RPM)
        rpm = super()._normalizedActionToRPM(action)

        return rpm
    

    def step(self, action):
        self.t += self.dt
        obs, reward, terminated, truncated, info = super().step(action) # calculates obs and reward after updating kinematic state
        
        p_ref, _, _ = self.target_state_generator.get(self.t)
        self.prev_vel = self.vel[0].copy()
        self.prev_act = self.last_clipped_action / self.MAX_RPM
        self.prev_dist = np.linalg.norm(p_ref - self.pos[0])

        return obs, reward, terminated, truncated, info
    

    def reset(self, seed=None, options=None):
        self.prev_vel = np.zeros(3)   # placeholder: BaseAviary.reset() calls _computeObs() internally

        super().reset(seed=seed, options=options)

        self.init_state_gen.reset(self.np_random)
        self.target_state_generator.reset(self.np_random)
        self.t = 0

        xyz, vel, rpy, ang_vel = self.init_state_gen.get()
        drone_id, client = self.DRONE_IDS[0], self.getPyBulletClient()
        p.resetBasePositionAndOrientation(drone_id, xyz.tolist(), p.getQuaternionFromEuler(rpy.tolist()), physicsClientId=client)
        p.resetBaseVelocity(drone_id, vel.tolist(), ang_vel.tolist(), physicsClientId=client)
        self._updateAndStoreKinematicInformation()

        p_ref, _, _ = self.target_state_generator.get(self.t)
        self.prev_vel = self.vel[0].copy()
        self.prev_act = np.full(4, self.HOVER_RPM / self.MAX_RPM)
        self.prev_dist = np.linalg.norm(p_ref - self.pos[0])

        return self._computeObs(), self._computeInfo()