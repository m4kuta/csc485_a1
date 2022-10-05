#!/usr/bin/env python3

import typing as T

from graphdep import Batch, GraphDepModel


def print_header(header: str, space: bool = True) -> None:
    if space:
        print()
    border = 80 * '='
    print(border, f'{header:^80}', border, sep='\n')


def do_eval(batch_iter: T.Iterable[Batch], model: GraphDepModel,
            desc: T.Optional[str] = None) -> T.Tuple[float, float, float]:
    uas_correct, las_correct, total, tree_sent, tot_sent = 0, 0, 0, 0, 0
    with tqdm(batch_iter, desc=desc, leave=False, disable=None, unit='batch',
              dynamic_ncols=True) as it:
        for batch in it:
            b_ucorr, b_lcorr, b_tot, b_trees = model.eval_batch(batch)
            uas_correct += b_ucorr
            las_correct += b_lcorr
            total += b_tot
            tree_sent += b_trees
            tot_sent += len(batch[4])
            it.set_postfix(LAS=f'{100 * las_correct / total:.1f}')
    return uas_correct / total, las_correct / total, tree_sent / tot_sent


if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    from pathlib import Path

    import torch
    from torch.utils.data import DataLoader

    from tqdm import tqdm

    import config as cfg
    from data import UDData

    argparser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    argparser.add_argument('--debug', action='store_true',
                           help='Enable debugging mode: only processes 10% of '
                                'data per epoch and disables background data '
                                'processing. Make sure not to use this flag '
                                "when you're ready to train your final model!")
    args = argparser.parse_args()

    print_header(f'INITIALIZING{" (debug mode)" if args.debug else ""}', False)
    if gpu := torch.cuda.is_available():
        print(f'Running on GPU: {torch.cuda.get_device_name()}.')
    else:
        print('Running on CPU.')
    torch.manual_seed(0)

    print('Reading data...', end=' ', flush=True)
    train, dev, test = UDData.read(*cfg.UD_CORPUS,
                                   fraction=0.1 if args.debug else 1)
    print('Done.')

    print('Initializing model...', end=' ', flush=True)
    model = GraphDepModel(cfg.HFTF_MODEL_NAME, len(train.deprels_s2i))
    if gpu:
        model = model.cuda()
    print('Done.', flush=True)

    print_header(f'TRAINING{" (debug mode)" if args.debug else ""}')
    train_dl = DataLoader(train, batch_size=cfg.BATCH_SIZE, shuffle=True,
                          num_workers=0 if args.debug else 2,
                          collate_fn=model.collate, pin_memory=gpu,
                          persistent_workers=not args.debug)
    dev_dl = DataLoader(dev, batch_size=cfg.BATCH_SIZE, shuffle=False,
                        num_workers=0 if args.debug else 2,
                        collate_fn=model.collate, pin_memory=gpu,
                        persistent_workers=not args.debug)
    weights_file = Path('weights-q2.pt')
    print(f'Best weights will be saved to {weights_file}.\n')

    barfmt = ('{l_bar}{bar}| %d/' + str(cfg.EPOCHS)
              + ' [{elapsed}<{remaining}{postfix}]')
    epoch_steps = len(train_dl) + len(dev_dl)
    best_las = 0.
    with tqdm(total=cfg.EPOCHS * epoch_steps, desc='Training', disable=None,
              unit='epoch', dynamic_ncols=True, bar_format=barfmt % 0) as pbar:
        for epoch in range(1, cfg.EPOCHS + 1):
            with tqdm(train_dl, desc=f'Epoch {epoch}', leave=False,
                      disable=None, unit='batch', dynamic_ncols=True) as it:
                for batch in it:
                    trn_loss = model.train_batch(batch)
                    it.set_postfix(loss=trn_loss)
                    pbar.update()
            dev_uas, dev_las, _ = do_eval(dev_dl, model, 'Validating')
            if best := dev_las > best_las:
                best_las = dev_las
                torch.save(model.state_dict(), str(weights_file))
            tqdm.write(f'Epoch {epoch:>2} validation LAS: {dev_las:.1%}'
                       f'{" (BEST!)" if best else  "        "} '
                       f'UAS: {dev_uas:.1%}')
            pbar.bar_format = barfmt % epoch

    print_header('TESTING')
    print('Restoring the best model weights found on the dev set...',
          end=' ', flush=True)
    model.load_state_dict(torch.load(str(weights_file)))
    print('Done.', flush=True)

    tst_dl = DataLoader(test, batch_size=cfg.BATCH_SIZE, shuffle=False,
                        num_workers=0 if args.debug else 2,
                        collate_fn=model.collate, pin_memory=gpu,
                        persistent_workers=not args.debug)
    tst_uas, tst_las, tst_trees = do_eval(tst_dl, model, 'Testing')
    print(f'Test LAS: {tst_las:.1%} UAS: {tst_uas:.1%} Trees: {tst_trees:.1%}')

    print_header('DONE')
