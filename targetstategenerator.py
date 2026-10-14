import numpy as np

class TargetStateGenerator:

    def reset(self):
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
        return self.xyz, np.zeros(3)
    
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
        return self.p_0 + t * self.v, self.v

    
class CircleTargetStateGenerator(TargetStateGenerator):

    def __init__(self, radius=1.0, speed=1.0):
        self.radius = radius
        self.speed = speed

    def reset(self):
        pass

    def get(self, t):
        x = self.radius * np.cos(self.speed * t)
        y = self.radius * np.sin(self.speed * t)
        z = 1.0

        vx = -self.radius * self.speed * np.sin(self.speed * t)
        vy = self.radius * self.speed * np.cos(self.speed * t)
        vz = 0.0

        return np.array([x, y, z]), np.array([vx, vy, vz])

class SawtoothTargetStateGenerator(TargetStateGenerator):

    def __init__(self, frequency=0.5, amplitude=np.array([1.0, 0.5, 0.0])):
        self.freq = frequency
        self.amp = amplitude

    def get(self, t):
        # Standard triangle wave formula: 2 * abs(2 * (t * f - floor(t * f + 0.5))) - 1
        def tri(time, f):
            return 2 * np.abs(2 * (time * f - np.floor(time * f + 0.5))) - 1
        
        # Derivative of triangle wave is a square wave
        def tri_deriv(time, f):
            return 4 * f * np.sign(np.cos(2 * np.pi * time * f))

        pos = np.array([t * 0.5, tri(t, self.freq) * self.amp[1], 1.0])
        vel = np.array([0.5, tri_deriv(t, self.freq) * self.amp[1], 0.0])
        
        return pos, vel