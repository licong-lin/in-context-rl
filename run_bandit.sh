python3 collect_data.py --env bandit --envs 100000 --H 200 --dim 5 --act_num 10 --controller LinUCB --var 1.5  --envs_eval 200



python3 train.py --env bandit --envs 100000 --H 200 --dim 5 --var 1.5 --act_num 10 --lr 0.001 --controller LinUCB --layer 8 --head 4 --shuffle  --imit LinUCB --num_epochs 2000 --embd 32  --act_type relu



python3 eval.py --env bandit --envs 100000 --H 200 --dim 5 --var 1.5 --act_num 10 --controller LinUCB --lr 0.0005 --layer 8 --head 4 --shuffle --epoch 2000 --n_eval 200  --imit LinUCB --embd 32   --act_type relu   

