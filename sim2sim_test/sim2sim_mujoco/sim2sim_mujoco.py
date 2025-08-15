# BSD 3-Clause License
# Copyright (c) 2025 Cyan Technologies Co., Ltd.
# All rights reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import time
import mujoco.viewer
import mujoco
import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation as R
import argparse
from legged_gym import LEGGED_GYM_ROOT_DIR

def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd

def load_config(config_file):
    with open(f"{LEGGED_GYM_ROOT_DIR}/sim2sim_test/sim2sim_mujoco/configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config["policy_path"] = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        config["xml_path"] = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        config["kps"] = np.array(config["kps"], dtype=np.float32)
        config["kds"] = np.array(config["kds"], dtype=np.float32)
        config["default_angles"] = np.array(config["default_angles"], dtype=np.float32)
        config["cmd_scale"] = np.array(config["cmd_scale"], dtype=np.float32)
        config["cmd_init"] = np.array(config["cmd_init"], dtype=np.float32)
    return config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="Config file name in the config folder")
    args = parser.parse_args()
    
    config = load_config(args.config_file)
    
    # Initialize variables
    action = np.zeros(config["num_actions"], dtype=np.float32)
    target_dof_pos = config["default_angles"].copy()
    obs = np.zeros(config["num_obs"], dtype=np.float32)
    counter = 0

    # Load robot model and policy
    m = mujoco.MjModel.from_xml_path(config["xml_path"])
    d = mujoco.MjData(m)
    m.opt.timestep = config["simulation_dt"]
    policy = torch.jit.load(config["policy_path"])

    with mujoco.viewer.launch_passive(m, d) as viewer:
        start_time = time.time()
        while viewer.is_running() and time.time() - start_time < config["simulation_duration"]:
            step_start = time.time()
            
            # Apply PD control
            tau = pd_control(
                target_dof_pos, 
                d.qpos[7:], 
                config["kps"], 
                np.zeros_like(config["kds"]), 
                d.qvel[6:], 
                config["kds"]
            )
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)
            
            # Policy inference at decimated rate
            if counter % config["control_decimation"] == 0:
                # Prepare observations
                qj = (d.qpos[7:] - config["default_angles"]) * config["dof_pos_scale"]
                dqj = d.qvel[6:] * config["dof_vel_scale"]
                omega = d.qvel[3:6] * config["ang_vel_scale"]
                quat = d.qpos[3:7][[1, 2, 3, 0]]
                rpy = R.from_quat(quat).as_euler('xyz').astype(np.float32)
                
                # Periodic signal generation
                phase = (counter * config["simulation_dt"]) % 0.8 / 0.8
                sin_cos_phase = np.array([np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)])
                
                # Build observation vector
                obs_sections = [
                    omega, rpy, config["cmd_init"] * config["cmd_scale"],
                    qj, dqj, action, sin_cos_phase
                ]
                obs = np.concatenate(obs_sections)
                
                # Run policy
                with torch.no_grad():
                    action = policy(torch.from_numpy(obs).unsqueeze(0).to(torch.float32)).numpy().squeeze()
                target_dof_pos = action * config["action_scale"] + config["default_angles"]
            
            counter += 1
            viewer.sync()
            
            # Maintain real-time simulation
            elapsed = time.time() - step_start
            if elapsed < config["simulation_dt"]:
                time.sleep(config["simulation_dt"] - elapsed)

if __name__ == "__main__":
    main()