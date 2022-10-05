#!/usr/bin/env python3

if __name__ == '__main__':
    import xml.etree.ElementTree as ET
    from pathlib import Path

    import torch
    from torch.utils.data import DataLoader

    from tqdm import tqdm

    from conllu import TokenList

    import config as cfg
    from data import UDData
    from graphdep import GraphDepModel

    if gpu := torch.cuda.is_available():
        print(f'Running on GPU: {torch.cuda.get_device_name()}.')
    else:
        print('Running on CPU.')
    torch.manual_seed(0)

    print('Reading data...', end=' ', flush=True)
    data_root = cfg.DATA_DIR / f'UD_{cfg.UD_CORPUS[0]}-{cfg.UD_CORPUS[1]}'
    deprel_s2i = {'root': 0}  # keep root deprel as 0 for simplicity
    deprel_i2s = ['root']
    for dep in ET.parse(data_root / 'stats.xml').getroot().iterfind('.//dep'):
        if (deprel := dep.attrib['name']) not in deprel_s2i:
            deprel_s2i[deprel] = len(deprel_s2i)
            deprel_i2s.append(deprel)
    dev = UDData(data_root / 'en_ewt-ud-dev.conllu', deprel_s2i)
    test = UDData(data_root / 'en_ewt-ud-test.conllu', deprel_s2i)
    print('Done.')

    weights_file = Path('weights-q2.pt')
    print(f'Loading model weights from {weights_file}...', end=' ', flush=True)
    model = GraphDepModel(cfg.HFTF_MODEL_NAME, len(dev.deprels_s2i))
    model.load_state_dict(torch.load(str(weights_file),
                                     map_location='cuda' if gpu else 'cpu'))
    # NOTE: there's a bug in PyTorch that causes a relatively harmless error
    #  message if pin_memory=True and num_workers>0. since pin_memory
    #  doesn't make a huge difference here, it's hard-set to False.
    #  The bug is fixed as of v1.9.1, so once that version is being used,
    #  we can revert to using pin_memory=gpu here.
    if gpu:
        model = model.cuda()
    model.eval()
    print('Done.', flush=True)

    dev_dl = DataLoader(dev, batch_size=cfg.BATCH_SIZE, shuffle=False,
                        num_workers=2, collate_fn=model.collate,
                        pin_memory=False, persistent_workers=True)
    tst_dl = DataLoader(test, batch_size=cfg.BATCH_SIZE, shuffle=False,
                        num_workers=2, collate_fn=model.collate,
                        pin_memory=False, persistent_workers=True)
    dev_out, tst_out = Path('dev.out.conllu'), Path('tst.out.conllu')
    print(f'Outputs will be saved to {dev_out} and {tst_out}.\n')

    barfmt = ('{l_bar}{bar}| %d/2 [{elapsed}<{remaining}{postfix}]')
    tot_steps = len(dev_dl) + len(tst_dl)
    with tqdm(total=tot_steps, desc='Running', disable=None, unit='epoch',
              dynamic_ncols=True, bar_format=barfmt % 1) as pbar:
        def run_and_save(dl: DataLoader, ds: UDData, filepath: Path,
                         desc: str = ''):
            with tqdm(dl, desc=desc, leave=False, disable=None, unit='batch',
                      dynamic_ncols=True) as it, filepath.open('w') as fout:
                for i, batch in enumerate(it):
                    i *= cfg.BATCH_SIZE
                    sentences = ds[i:i + cfg.BATCH_SIZE][0]
                    batch = model.transfer_batch(batch)
                    pred_heads, pred_labels, _, _ = model._predict_batch(batch)
                    for sent, pheads, plabels in zip(sentences,
                                                     pred_heads.cpu(),
                                                     pred_labels.cpu()):
                        tl = TokenList(
                            [{'id': j, 'form': w, 'lemma': '_', 'upos': '_',
                              'xpos': '_', 'feats': '_', 'head': h.item(),
                              'deprel': deprel_i2s[l.item()], 'deps': '_',
                              'misc': '_'}
                             for j, (w, h, l) in
                             enumerate(zip(sent, pheads, plabels), start=1)])
                        fout.write(tl.serialize())
                    pbar.update()

        run_and_save(dev_dl, dev, dev_out, 'Dev set')
        pbar.bar_format = barfmt % 2
        run_and_save(tst_dl, test, tst_out, 'Test set')
