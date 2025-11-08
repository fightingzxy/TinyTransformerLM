#!/usr/bin/env bash

python -m src.train --exp_name baseline --epochs 2 --seed 42

python -m src.train --exp_name head2 --n_head 2 --epochs 2 --seed 42
python -m src.train --exp_name head4 --n_head 4 --epochs 2 --seed 42

python -m src.train --exp_name d128  --d_model 128 --epochs 2 --seed 42
python -m src.train --exp_name d256  --d_model 256 --epochs 2 --seed 42

python -m src.train --exp_name layer2 --n_layer 2 --epochs 2 --seed 42
python -m src.train --exp_name layer3 --n_layer 3 --epochs 2 --seed 42

python -m src.train --exp_name no_pe --no_pe --epochs 2 --seed 42

