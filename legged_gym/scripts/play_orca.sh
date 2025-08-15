export LD_LIBRARY_PATH=/home/cbf/miniconda3/envs/cyan-rl/lib

python play.py --task=orca \
                --checkpoint -1 \
                --num_envs 32 \
                --sim_device 'cpu' \
                --rl_device 'cpu' \

