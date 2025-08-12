CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc-per-node=4 train.py --config=configs/train/contrastive_finetune_inc_1.yaml --dtype=bf16 --task_num=1 --use_kd=0
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc-per-node=4 train.py --config=configs/train/contrastive_finetune_inc_2.yaml --dtype=bf16 --task_num=2 --use_kd=0
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc-per-node=4 train.py --config=configs/train/contrastive_finetune_inc_3.yaml --dtype=bf16 --task_num=3 --use_kd=0
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc-per-node=4 train.py --config=configs/train/contrastive_finetune_inc_4.yaml --dtype=bf16 --task_num=4 --use_kd=0
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc-per-node=4 train.py --config=configs/train/contrastive_finetune_inc_5.yaml --dtype=bf16 --task_num=5 --use_kd=0


CUDA_VISIBLE_DEVICES=0 python eval/eval_retrieval.py --new_model_name=ckpts/nomic-marco_nq_hotpot_fever_fiqa/epoch_0_model --model_name=ckpts/nomic-marco_nq_hotpot_fever_fiqa-all/epoch_0_model --task_num=5