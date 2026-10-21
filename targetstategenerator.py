import numpy as np

class TargetStateGenerator:

    def reset(self, rng = None):
        pass

    def get(self, t):
        raise NotImplementedError
    

class StaticTargetStateGenerator(TargetStateGenerator):

    def __init__(
        self, 
        xyz=np.array([0,0,1.0]),  
        min_dist=0.5, 
        max_dist=2.0, 
        min_z=0.3, 
        max_z=1.5,
        randomize=False, 
        rng = None
    ):
        self.xyz = np.array(xyz, dtype=float)
        self.rng = None
        self.randomize = randomize
        self.rng = rng if rng is not None else np.random.default_rng()
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.min_z = min_z
        self.max_z = max_z

    def reset(self, rng = None):
        if self.randomize:
            self.rng = rng if rng is not None else np.random.default_rng()
            # Random horizontal distance and direction
            dist = self.rng.uniform(self.min_dist, self.max_dist)
            angle = self.rng.uniform(-np.pi, np.pi)   # full 360° coverage
            self.xyz = np.array([
                dist * np.cos(angle),
                dist * np.sin(angle),
                self.rng.uniform(self.min_z, self.max_z),
            ])

    def get(self, t):
        return self.xyz, np.zeros(3), np.zeros(3)
    
class LinearTargetStateGenerator(TargetStateGenerator):

    def __init__(
        self, 
        p_0=np.array([0,0,1.0]), 
        v=np.array([1.0,0,0]), 
        randomize=False, 
        rng = None
    ):
        self.p_0 = p_0
        self.v = v  # m/s
        self.rng = rng if rng is not None else np.random.default_rng()
        self.randomize = randomize

    def reset(self, rng = None):
        if self.randomize:
           self.rng = rng if rng is not None else np.random.default_rng()
           self.p_0 = np.array([
               self.rng.uniform(low=-1.0, high=1.0),
               self.rng.uniform(low=-1.0, high=1.0),
               1.0
           ])
           self.v = np.array([
               self.rng.uniform(low=0, high=1.0),
               self.rng.uniform(low=-1.0, high=1.0),
               0
           ])

    def get(self, t):
        return self.p_0 + t * self.v, self.v, np.zeros(3)

    
class CircleTargetStateGenerator(TargetStateGenerator):

    def __init__(self, radius=1.0, speed=1.0):
        self.radius = radius
        self.speed = speed

    def get(self, t):
        w = self.speed
        c, s = np.cos(w * t), np.sin(w * t)

        pos = np.array([self.radius * c, self.radius * s, 1.0])
        vel = np.array([-self.radius * w * s, self.radius * w * c, 0.0])
        acc = np.array([-self.radius * w**2 * c, -self.radius * w**2 * s, 0.0])  # centripetal: -ω² (pos - center)

        return pos, vel, acc 