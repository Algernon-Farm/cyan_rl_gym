export LD_LIBRARY_PATH=/home/cbf/miniconda3/envs/cyan-rl/lib

python train.py --task=orca \
                --num_envs 4096 \
                --sim_device 'cuda:0' \
                --rl_device 'cuda:0' \
                --run_name='orca' \
                --headless \
