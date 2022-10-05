#!/usr/bin/env python3

from __future__ import annotations

import re
import typing as T
import xml.etree.ElementTree as ET
from enum import IntFlag, auto
from multiprocessing import Pool, cpu_count
from pathlib import Path

from torch.utils.data import Dataset

from conllu import parse_token_and_metadata

import config as cfg
from graphalg import is_projective


class SentIncl(IntFlag):
    NONPROJ = auto()
    PROJ = auto()
    BOTH = NONPROJ | PROJ


class UDData(Dataset):
    split_sent = re.compile('\n\n')

    def __init__(self, filepath: Path, deprels: T.Mapping[str, int], *,
                 include: SentIncl = SentIncl.BOTH, fraction: float = 1.):
        if not include:
            raise ValueError(f'must include some sentences!')
        if filepath.suffix == '.bz2':
            from bz2 import open as file_open
        else:
            file_open = open
        with file_open(filepath, 'rt') as data_in:
            data = data_in.read()
        sentences = [sent for sent in self.split_sent.split(data) if sent]
        if 0 < fraction < 1:
            sentences = sentences[:int(len(sentences) * fraction)]
        self.deprels_s2i = deprels

        proc = min(cpu_count(), 2)
        with Pool(proc) as pool:
            sentences = pool.map(self.parse_etc, sentences)
        sentences = [s for s in sentences if (s[-1] + 1) & include]
        data = [list(t) for t in zip(*sentences)]
        self.forms, self.heads, self.deprels, self.projective = data

    def parse_etc(self, s: str) -> T.Tuple[T.List[str, ...], T.Tuple[int, ...],
                                           T.List[int, ...], bool]:
        sen = parse_token_and_metadata(s).filter(id=lambda x: type(x) is int)
        forms, heads, deprels = zip(*[(t['form'], t['head'],
                                       self.deprels_s2i[t['deprel']])
                                      for t in sen])
        return forms, heads, deprels, is_projective(heads)

    def __getitem__(self, item: int) \
            -> T.Tuple[T.Tuple[str, ...], T.Tuple[int, ...],
                       T.List[int, ...], bool]:
        return (self.forms[item], self.heads[item], self.deprels[item],
                self.projective[item])

    def __len__(self) -> int:
        return len(self.forms)

    @classmethod
    def read(cls, language: str, treebank: str, *, fraction: float = 1.) \
            -> T.Tuple[T.Optional[UDData], T.Optional[UDData],
                       T.Optional[UDData]]:
        root = cfg.DATA_DIR / f'UD_{language}-{treebank}'
        deprel_s2i = {'root': 0}  # keep root deprel as 0 for simplicity
        for dep in ET.parse(root / 'stats.xml').getroot().iterfind('.//dep'):
            if (deprel := dep.attrib['name']) not in deprel_s2i:
                deprel_s2i[deprel] = len(deprel_s2i)
        datasets = []
        for split in ['train', 'dev', 'test']:
            if files := list(root.glob(f'*{split}.conll*')):
                assert len(files) == 1
                datasets.append(cls(files[0], deprel_s2i, fraction=fraction))
            else:
                datasets.append(None)

        return tuple(datasets)


