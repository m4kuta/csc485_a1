#!/usr/bin/env python3


if __name__ == '__main__':
    from tqdm import tqdm

    from data import UDData

    import config as cfg

    print('Reading data...', end=' ', flush=True)
    train, dev, test = UDData.read(*cfg.UD_CORPUS)
    print('Done.', flush=True)

    for ds in ['train', 'dev', 'test']:
        data = globals()[ds]
        projective = 0
        for datum in tqdm(data):
            projective += datum[-1]
        print(f'{ds}: {projective}/{len(data)} ({projective / len(data):.1%})')
