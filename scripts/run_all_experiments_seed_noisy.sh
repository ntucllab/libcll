strategy=$1
tp=$2
seed=$3
cuda=0
strategies=($strategy)
types=(${tp})
lrs=("1e-3" "5e-4" "1e-4" "5e-5" "1e-5")
lrs=($(cat "logs/${strategy}/${strategy}-${tp}-noisy.txt"))
datasets=("mnist" "kmnist" "fmnist" "cifar10" "micro_imagenet10")
for strategy in ${strategies[@]}; do
    i=0
    for dataset in ${datasets[@]}; do
         
        if [[ $dataset == "mnist" ]] || [[ $dataset == "kmnist" ]] || [[ $dataset == "fmnist" ]] || [[ $dataset == "yeast" ]] || [[ $dataset == "texture" ]] || [[ $dataset == "control" ]] || [[ $dataset == "dermatology" ]]; then
            models=("Linear" "MLP")
            models=("MLP")
        elif [[ $dataset == "cifar10" ]] || [[ $dataset == "cifar20" ]] || [[ $dataset == "clcifar10" ]] || [[ $dataset == "clcifar20" ]] || [[ $dataset == "micro_imagenet10" ]] || [[ $dataset == "micro_imagenet20" ]] || [[ $dataset == "clmicro_imagenet10" ]] || [[ $dataset == "clmicro_imagenet20" ]]; then
            models=("ResNet34" "DenseNet")
       	    models=("ResNet34")
        fi
        valid_type="Accuracy"
        multi=1
        for noise in "0.1" "0.2" "0.5"; do
            lr=${lrs[$i]}
            i=$(($i+1))
            for t in ${types[@]}; do
                for model in ${models[@]}; do
                    echo "scripts/noise.sh ${cuda} ${strategy} ${t} ${model} ${dataset} ${valid_type} ${multi} noisy ${lr} ${seed} ${noise}"
                    scripts/noise.sh ${cuda} ${strategy} ${t} ${model} ${dataset} ${valid_type} ${multi} noisy ${lr} ${seed} ${noise}
                done
            done
        done
    done
done

