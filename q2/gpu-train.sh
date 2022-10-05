#!/usr/bin/env bash

srun -p csc485 --gres gpu -c 2 python3 train.py
