import numpy as np

class TrajectoryGenerator:
    """
    Base class for time-varying target point generators.
    """
    def reset(self):
        pass

    def get(self, t):
        """
        Returns:
            position (3,)
            velocity (3,)
        """
        raise NotImplementedError
    

class StaticTrajectory(TrajectoryGenerator):
    """
    Stationary target point. On each reset, samples a new target at a random
    distance and direction from the drone's spawn.
    """
    def __init__(self, xyz=np.array([0,0,1.0]), randomize=False, min_dist=0.5, max_dist=2.0, min_z=0.3, max_z=1.5):
        self.xyz = np.array(xyz, dtype=float)
        self.randomize = randomize
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.min_z = min_z
        self.max_z = max_z

    def reset(self):
        if self.randomize:
            # Random horizontal distance and direction
            dist = np.random.uniform(self.min_dist, self.max_dist)
            angle = np.random.uniform(-np.pi, np.pi)   # full 360° coverage
            self.xyz = np.array([
                dist * np.cos(angle),
                dist * np.sin(angle),
                np.random.uniform(self.min_z, self.max_z),
            ])

    def get(self, t):
        return self.xyz, np.zeros(3)
    
class LinearTrajectory(TrajectoryGenerator):
    """
    Target point moving at a constant velocity v from p_0. On each reset, samples p_0 at a random
    distance and direction from the drone's spawn, and random v.
    """
    def __init__(self, p_0=np.array([0,0,1.0]), v=np.array([1.0,0,0]), randomize=False):
        self.p_0 = p_0
        self.v = v  # m/s
        self.randomize = randomize

    def reset(self):
        if self.randomize:
           self.p_0 = np.array([
               np.random.uniform(low=-1.0, high=1.0),
               np.random.uniform(low=-1.0, high=1.0),
               1.0
           ])
           self.v = np.array([
               np.random.uniform(low=0, high=1.0),
               np.random.uniform(low=-1.0, high=1.0),
               0
           ])

    def get(self, t):
        return self.p_0 + t * self.v, self.v

    
class CircleTrajectory(TrajectoryGenerator):
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

class ZigZagTrajectory(TrajectoryGenerator):
    """
    Generates a path consisting of sharp turns between random waypoints.
    The trajectory moves at a constant speed 'v' between vertices.
    """
    def __init__(self, speed=1.5, bounds=np.array([2.0, 2.0, 1.0]), n_points=20):
        
        self.speed = speed
        self.bounds = bounds
        self.n_points = n_points
        self.waypoints = []
        self.times = []
        self.reset()

    def reset(self):
        # Generate random waypoints within bounds
        # Starting point near origin-ish
        pts = [np.array([0, 0, 1.0])]
        
        for _ in range(self.n_points):
            next_pt = np.random.uniform(-self.bounds, self.bounds)
            # Ensure Z stays above floor
            next_pt[2] = np.random.uniform(0.5, self.bounds[2] + 0.5)
            pts.append(next_pt)
        
        self.waypoints = np.array(pts)
        
        # Calculate arrival times at each waypoint based on constant speed
        self.times = [0.0]
        for i in range(len(self.waypoints) - 1):
            dist = np.linalg.norm(self.waypoints[i+1] - self.waypoints[i])
            self.times.append(self.times[-1] + dist / self.speed)
        
        self.times = np.array(self.times)

    def get(self, t):
        # Handle time beyond the last waypoint
        if t >= self.times[-1]:
            return self.waypoints[-1], np.zeros(3)
        
        # Find the segment we are currently in
        idx = np.searchsorted(self.times, t) - 1
        idx = max(0, idx)
        
        t_start = self.times[idx]
        t_end = self.times[idx+1]
        p_start = self.waypoints[idx]
        p_end = self.waypoints[idx+1]
        
        # Interpolate position
        alpha = (t - t_start) / (t_end - t_start)
        pos = p_start + alpha * (p_end - p_start)
        
        # Calculate velocity (direction * speed)
        vel = (p_end - p_start) / (t_end - t_start)
        
        return pos, vel
    
class SawtoothTrajectory(TrajectoryGenerator):
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