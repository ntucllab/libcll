export CUDA_VISIBLE_DEVICES=$1

strategy=$2
tp=$3
model=$4
dataset=$5
valid_type=$6
num_cl=$7
transition_matrix=$8
lr=$9
seed=${10}
noise=${11}

output_dir="logs/${strategy}/${dataset}-multi_label_${num_cl}-${transition_matrix}_${noise}/${strategy}-${tp}-${model}-${dataset}-${lr}-${seed}"
python scripts/train.py \
    --do_train \
    --do_predict \
    --strategy ${strategy} \
    --type ${tp} \
    --model ${model} \
    --dataset ${dataset} \
    --lr ${lr} \
    --batch_size 256 \
    --augment \
    --valid_type ${valid_type} \
    --output_dir ${output_dir} \
    --num_cl ${num_cl} \
    --transition_matrix ${transition_matrix}\
    --epoch 300 \
    --noise ${noise} \
    --seed ${seed} \

rm ${output_dir}/*.ckpt
