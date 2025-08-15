# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-FileCopyrightText: Copyright (c) 2025 Cyan Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
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

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class OrcaRoughCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.85] # x,y,z [m]       
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'lleg_joint1': 0.,
            'lleg_joint2': 0.,
            'lleg_joint3': 0.,
            'lleg_joint4': 0.,
            'lleg_joint5': 0.,
            'lleg_joint6': 0.,

            'rleg_joint1': 0.,
            'rleg_joint2': 0.,
            'rleg_joint3': 0.,
            'rleg_joint4': 0.,
            'rleg_joint5': 0.,
            'rleg_joint6': 0.,
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        stiffness = {  
                        'leg_joint1': 75.,
                        'leg_joint2': 50.,
                        'leg_joint3': 50.,
                        'leg_joint4': 75.,
                        'leg_joint5': 30.,
                        'leg_joint6': 5 #0.,
                    }  # [N*m/rad]

        damping =   {  
                        'leg_joint1': 3.,
                        'leg_joint2': 3.,
                        'leg_joint3': 3., # 5.
                        'leg_joint4': 3.,
                        'leg_joint5': 2.,
                        'leg_joint6': 1 #5.,
                    }  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 5

        
        
    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/orca_description/urdf/orca_description_12dof.urdf'
        name = "orca"
        foot_name = 'leg_link6'
        terminate_after_contacts_on = []
        flip_visual_attachments = False
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.78
        
        class scales( LeggedRobotCfg.rewards.scales ):
            tracking_lin_vel = 1.0 
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -1.0
            base_height = -10.0
            dof_acc = -2.5e-7 * 1.5
            dof_vel = -1e-3 * 1.5
            base_acc = -1e-1
            feet_air_time = 0.2
            collision = 0.0
            action_rate = -0.01 * 1.5
            dof_pos_limits = -5.0 
            alive = 0.15
            hip_pos = -1.5 
            contact_no_vel = -0.2
            feet_swing_height = -20.0 * 1.2
            contact = 0.36


class OrcaRoughCfgPPO( LeggedRobotCfgPPO ):
    
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_orca'
        max_iterations = 10000

    class algorithm( LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01



  