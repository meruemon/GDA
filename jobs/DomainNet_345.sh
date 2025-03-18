echo "start..."

source=$1
target=$2
lambda_a=$3
lambda_b=$4
lambda_c=$5
mode=$6


dataset="DomainNet"
data_dir="./data/DomainNet"
out_dir=MemSAC_${source}${target}
nClasses=345
BatchSize=32
queue_size=48000

python3 train.py --dataset ${dataset} --source ${source} --target ${target} --lr 0.03 --out_dir ${out_dir} --max_iteration 100000 \
--batch_size ${BatchSize} --data_dir ${data_dir} --total_classes ${nClasses} --multi_gpu 0 --test-iter 5000 --queue_size ${queue_size} \
--adv-coeff 1. --sim-coeff 0.1 --lambda_a ${lambda_a} --lambda_b ${lambda_b} --lambda_c ${lambda_c} --mode ${mode}
