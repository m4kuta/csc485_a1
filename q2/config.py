#!/usr/bin/env python3

from pathlib import Path

# set MODEL_DIR to None in order to run this outside of teach.cs
MODEL_DIR = Path('/u/csc485h/fall/pub/transformers')
# and adjust DATA_DIR to wherever you've copied the corpora directory to
DATA_DIR = Path('/u/csc485h/fall/pub/a1/corpora')

ARC_DIM = 256
LABEL_DIM = 64
BATCH_SIZE = 32
DROPOUT = 0.1
EPOCHS = 10
LR = 2e-3
HFTF_MODEL_NAME = 'bert-base-cased'
UD_CORPUS = ('English', 'EWT')
