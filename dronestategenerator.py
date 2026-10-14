import numpy as np

class DroneStateGenerator():
    def __init__(
        self, 
        init_xyz = np.array([0, 0, 1.0]), 
        init_rpy = np.array([0, 0, 0]),
        init_vel = np.array([0, 0, 0]),
        init_ang_vel = np.array([0, 0, 0]),
        randomize = False,
        rng = None
    ):
        self.init_xyz = init_xyz
        self.init_rpy = init_rpy
        self.init_vel = init_vel
        self.init_ang_vel = init_ang_vel
        self.rng = rng if rng is not None else np.random.default_rng()
        self.randomize = randomize

    def get(self):

        if self.randomize:
            return (
                self.init_xyz, 
                self.rng.uniform([-2, -2, -1], [2, 2, 1]), # Linear velocity
                self.rng.uniform([-0.44, -0.44, -np.pi], [0.44, 0.44, np.pi]), # roll ±25°, pitch ±25°, yaw ±180°
                self.rng.uniform([-1, -1, -1], [1, 1, 1]) # Angular velocity: ±1 rad/s
            )
        else:
            return (
                self.init_xyz,
                self.init_vel,
                self.init_rpy,
                self.init_ang_vel
            )  

        
    def reset(self, rng = None):
        if self.randomize:
            self.rng = rng if rng is not None else np.random.default_rng()