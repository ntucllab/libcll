strategy=$1
tp=$2
seed=$3
cuda=$4
strategies=($strategy)
types=(${tp})
lrs=("1e-3" "5e-4" "1e-4" "5e-5" "1e-5")
lrs=($(cat "logs/${strategy}/${strategy}-${tp}-multi-hard.txt"))
datasets=("micro_imagenet10" "micro_imagenet20" "clmicro_imagenet10" "clmicro_imagenet20" "aclmicro_imagenet10" "aclmicro_imagenet20")
for strategy in ${strategies[@]}; do
    for ((i=0; i < 6; i++)); do
        lr=${lrs[$i]}
        dataset=${datasets[$i]}
        if [[ $dataset == "mnist" ]] || [[ $dataset == "kmnist" ]] || [[ $dataset == "fmnist" ]] || [[ $dataset == "yeast" ]] || [[ $dataset == "texture" ]] || [[ $dataset == "control" ]] || [[ $dataset == "dermatology" ]]; then
            models=("Linear" "MLP")
            models=("MLP")
        elif [[ $dataset == "cifar10" ]] || [[ $dataset == "cifar20" ]] || [[ $dataset == "clcifar10" ]] || [[ $dataset == "clcifar20" ]] || [[ $dataset == "micro_imagenet10" ]] || [[ $dataset == "micro_imagenet20" ]] || [[ $dataset == "clmicro_imagenet10" ]] || [[ $dataset == "clmicro_imagenet20" ]] || [[ $dataset == "aclcifar10" ]] || [[ $dataset == "aclcifar20" ]] || [[ $dataset == "aclmicro_imagenet10" ]] || [[ $dataset == "aclmicro_imagenet20" ]]; then
            models=("ResNet34" "DenseNet")
       	    models=("ResNet34")
        fi
        valid_type="Accuracy"
        multi=3
        for t in ${types[@]}; do
            for model in ${models[@]}; do
                echo "scripts/seed.sh ${cuda} ${strategy} ${t} ${model} ${dataset} ${valid_type} ${multi} uniform ${lr} ${seed}"
                scripts/seed.sh ${cuda} ${strategy} ${t} ${model} ${dataset} ${valid_type} ${multi} uniform ${lr} ${seed}
            done
        done
    done
done

