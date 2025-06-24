strategy=$1
tp=$2
lrs=("1e-3" "5e-4" "1e-4" "5e-5" "1e-5")
echo "Start"
for ((i=0; i < ${#lrs[@]}; i++)); do
	lr=${lrs[$i]}
	c=0
	echo scripts/run_all_experiments_lr_multi.sh ${strategy} ${tp} ${lr} ${c}
	scripts/run_all_experiments_lr_multi.sh ${strategy} ${tp} ${lr} ${c} & 
done
wait
echo "End"

echo "Start"
for ((i=0; i < 3; i++)); do
	c=0
	echo scripts/run_all_experiments_lr_multi_hard.sh ${strategy} ${tp} ${i} ${c}
	scripts/run_all_experiments_lr_multi_hard.sh ${strategy} ${tp} ${i} ${c} & 
done
wait
echo "End"

echo "Start"
for ((i=3; i < 6; i++)); do
	c=0
	echo scripts/run_all_experiments_lr_multi_hard.sh ${strategy} ${tp} ${i} ${c}
	scripts/run_all_experiments_lr_multi_hard.sh ${strategy} ${tp} ${i} ${c} & 
done
wait
echo "End"

output_dir="logs/${strategy}/${strategy}-${tp}-multi.txt"

seeds=("1207" "9213" "17" "33")

python scripts/format_multi.py ${strategy}-${tp} ${output_dir} 0
echo "Start"
for seed in ${seeds[@]}; do
	echo scripts/run_all_experiments_seed_multi.sh ${strategy} ${tp} ${seed} ${c}
	scripts/run_all_experiments_seed_multi.sh ${strategy} ${tp} ${seed} ${c} &
done
wait
echo "End"
