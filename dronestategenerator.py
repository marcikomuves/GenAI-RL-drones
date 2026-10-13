import numpy as np

class DroneStateGenerator():
    def __init__(self, 
                 init_xyz = np.array([0, 0, 1.0]), 
                 init_rpy = np.array([0, 0, 0]),
                 init_vel = np.array([0, 0, 0]),
                 init_ang_vel = np.array([0, 0, 0]),
                 randomize = False,
                 seed = None
                 ):
        self.init_xyz = init_xyz
        self.init_rpy = init_rpy
        self.init_vel = init_vel
        self.init_ang_vel = init_ang_vel
        self.randomize = randomize

        self.rng = np.random.default_rng(seed)

    def sample_initial_conditions(self):
        """
        Sample a randomized initial condition for one episode.
        """
        
        xyz = self.init_xyz
        vel = self.init_vel if not self.randomize else self.rng.uniform([-2, -2, -1], [2, 2, 1]) # linear velocity: ±2 m/s
        ang_vel = self.init_ang_vel if not self.randomize else self.rng.uniform([-1, -1, -1], [1, 1, 1]) # Angular velocity: ±1 rad/s
        rpy = self.init_rpy if not self.randomize else self.rng.uniform([-0.44, -0.44, -np.pi], [0.44, 0.44, np.pi]) # roll ±25°, pitch ±25°, yaw ±180°

        return xyz, vel, rpy, ang_vel