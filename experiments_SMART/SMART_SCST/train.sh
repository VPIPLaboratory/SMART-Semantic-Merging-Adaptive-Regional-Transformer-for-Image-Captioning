CUDA_VISIBLE_DEVICES=0 python main.py --folder ./experiments_SMART/SMART_SCST/ --resume 20

# CUDA_VISIBLE_DEVICES=0, 1 python -m torch.distributed.launch --master_port=3142 --nproc_per_node=2 main_multi_gpu.py --folder ./experiments_SMART/SMART_SCST/ --resume 16