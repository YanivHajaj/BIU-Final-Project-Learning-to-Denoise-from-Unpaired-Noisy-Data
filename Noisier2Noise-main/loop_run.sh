
for i in {1..5}
do
    python3 test.py --exp_num 6 --n_epochs 200 --gpu_num 0 --dataset Set20 --aver_num $i
done
