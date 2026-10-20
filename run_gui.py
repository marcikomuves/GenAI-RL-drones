import sys, os
import argparse
import numpy as np
import time
import torch
import pybullet as p
from gym_pybullet_drones.utils.enums import ActionType, DroneModel, Physics
from gym_pybullet_drones.utils.utils import sync
from gym_pybullet_drones.utils.Logger import Logger
from targetaviary import TargetAviary
from targetstategenerator import *
from dronestategenerator import DroneStateGenerator
from sac import Actor as SACActor

def draw_axes():
    p.addUserDebugLine([0,0,0], [1,0,0], [1,0,0], 3, 0)
    p.addUserDebugLine([0,0,0], [0,1,0], [0,1,0], 3, 0)
    p.addUserDebugLine([0,0,0], [0,0,1], [0,0,1], 3, 0)

DRONE_MODEL = DroneModel.CF2X             
ACT = ActionType.RPM
PHYSICS = Physics.PYB
SIMULATION_FREQ_HZ = 240
CONTROL_FREQ_HZ = 48
INIT_XYZ = np.array([0, 0, 1.0])
MAX_STEPS = 50_000
RECORD = False

# Force Windows High-DPI scaling awareness before creating the GUI
if sys.platform == "win32":
    import ctypes
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1) # 1 = Process_System_DPI_Aware
    except Exception:
        pass

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-name", required=True)
    args = parser.parse_args()

    # Create env
    env = TargetAviary(
        target_state_generator=CircleTargetStateGenerator(),
        init_state_generator=DroneStateGenerator(init_xyz=INIT_XYZ, randomize=False),
        act=ACT, 
        max_t = 20,
        physics=PHYSICS, 
        drone_model=DRONE_MODEL, 
        pyb_freq=SIMULATION_FREQ_HZ, 
        ctrl_freq=CONTROL_FREQ_HZ,
        gui=True,
        record=RECORD
    )

    # Hide all GUI panels
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    actor = SACActor(act_space=env.action_space, obs_space=env.observation_space).to(device)

    if os.path.exists(args.policy_name):
        actor.load_state_dict(torch.load(args.policy_name, map_location=device))
    else:
        print("Policy path does not exist.")
        sys.exit(-1)

    p.resetDebugVisualizerCamera(
        cameraDistance=5.0,
        cameraYaw=0,
        cameraPitch=-89.999,
        cameraTargetPosition=[0, 0, 1.0],
    )

    actor.eval()
    obs, _ = env.reset()
    draw_axes()
    step_counter = 0
    start = time.time()
    for i in range(MAX_STEPS):
        #obs_tensor = torch.Tensor(obs.reshape(env.observation_space.shape)).to(device)
        with torch.no_grad():
            _, _, action_mean = actor.get_action(torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
        next_obs, _, terminated, truncated, _ = env.step(action_mean[0].cpu().numpy())
        step_counter += 1
        done = terminated or truncated
        if not done:
            obs = next_obs 
        else:
            obs, _ = env.reset()
            draw_axes()
            start = time.time()
            step_counter = 0

        sync(step_counter, start, env.CTRL_TIMESTEP)
        #drone_state, _ = env._getDroneStateVector(0), None
        #p.resetDebugVisualizerCamera(
        #    cameraDistance=2.0,
        #    cameraYaw=-90,
        #    cameraPitch=-30,
        #    cameraTargetPosition=drone_state[:3],
        #)

