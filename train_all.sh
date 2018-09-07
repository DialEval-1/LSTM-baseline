#!/usr/bin/env bash

python train_dch.py --category TS &&
python train_dch.py --category TA &&
python train_dch.py --category CS &&
python train_dch.py --category CA &&
python train_dch.py --category HA;
