set -e -x
cuda_gpu=1
python -u train.py -d tianchi -b max -r gru -copy 0 -cross 0 -tasks '0,1,2,3' -c ${cuda_gpu} -mix 0 -dim 100 -mtr 1 -wd 0.00005 -res 0 -lr 0.001
