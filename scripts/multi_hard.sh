strategy=$1
tp=$2
lrs=("1e-3" "5e-4" "1e-4" "5e-5" "1e-5")

echo "Start"
for ((i=6; i < 10; i++)); do
	c=$(($i%2))
	echo scripts/run_all_experiments_lr_multi_hard.sh ${strategy} ${tp} ${i} ${c}
	scripts/run_all_experiments_lr_multi_hard.sh ${strategy} ${tp} ${i} ${c} & 
done
wait
echo "End"

echo "Start"
for ((i=10; i < 12; i++)); do
	c=$(($i%2))
	echo scripts/run_all_experiments_lr_multi_hard.sh ${strategy} ${tp} ${i} ${c}
	scripts/run_all_experiments_lr_multi_hard.sh ${strategy} ${tp} ${i} ${c} & 
done
wait
echo "End"
output_dir="logs/${strategy}/${strategy}-${tp}-multi-hard.txt"

seeds=("1207" "9213" "17" "33")

python scripts/format_multi.py ${strategy}-${tp} ${output_dir} 1
echo "Start"
for ((i=0; i < 2; i++)); do
	seed=${seeds[$i]}
	c=$(($i%2))
	echo scripts/run_all_experiments_seed_multi_hard.sh ${strategy} ${tp} ${seed} ${c}
	scripts/run_all_experiments_seed_multi_hard.sh ${strategy} ${tp} ${seed} ${c} &
done
wait
echo "End"

echo "Start"
for ((i=2; i < 4; i++)); do
	seed=${seeds[$i]}
	c=$(($i%2))
	echo scripts/run_all_experiments_seed_multi_hard.sh ${strategy} ${tp} ${seed} ${c}
	scripts/run_all_experiments_seed_multi_hard.sh ${strategy} ${tp} ${seed} ${c} &
done
wait
echo "End"
