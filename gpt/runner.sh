# for sched in cosine linear wsd invsqrt constant; do
#     python train.py --max_stories 100000 --batch_size 64 --d_model 256 --num_layers 4 --rope --rmsnorm --scheduler $sched --run_name gpt_sched_$sched --epochs 3
# done

# Without LR scaling (same LR)
for bs in 32 64; do
    python train.py --epochs 2 --batch_size $bs --lr 3e-4 --run_name bs${bs}_fixedlr
done
python train.py --epochs 2 --batch_size 64 --grad_accum_steps 2 --lr 3e-4 --run_name bs128_fixedlr
python train.py --epochs 2 --batch_size 64 --grad_accum_steps 4 --lr 3e-4 --run_name bs256_fixedlr

# With LR scaling
python train.py --epochs 2 --batch_size 32 --lr 1.5e-4 --run_name bs32_scaledlr
python train.py --epochs 2 --batch_size 64 --lr 3e-4 --run_name bs64_scaledlr
python train.py --epochs 2 --batch_size 64 --grad_accum_steps 2 --lr 6e-4 --run_name bs128_scaledlr
python train.py --epochs 2 --batch_size 64 --grad_accum_steps 4 --lr 1.2e-3 --run_name bs256_scaledlr