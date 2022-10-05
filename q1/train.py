#!/usr/bin/env python3

from itertools import islice
from pathlib import Path
from sys import stdout

import click
import torch
from tqdm import tqdm

from data import load_and_preprocess_data
from model import Config, ParserModel

@click.command()
@click.option('--debug', is_flag=True)
def main(debug):
    """Main function

    Args:
    debug :
        whether to use a fraction of the data. Make sure not to use this flag
        when you're ready to train your model for real!
    """
    print(80 * '=')
    print(f'INITIALIZING{" debug mode" if debug else ""}')
    print(80 * '=')
    torch.manual_seed(1234)
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        torch.cuda.manual_seed_all(1234)
        print(f'Running on GPU: {torch.cuda.get_device_name()}.')
    else:
        print('Running on CPU.')
    config = Config()
    data = load_and_preprocess_data(max_batch_size=config.batch_size,
                                    transition_cache=0 if debug else None)
    transducer, word_embeddings, train_data = data[:3]
    dev_sents, dev_arcs = data[3:5]
    test_sents, test_arcs = data[5:]
    config.n_word_ids = len(transducer.id2word) + 1  # plus null
    config.n_tag_ids = len(transducer.id2tag) + 1
    config.n_deprel_ids = len(transducer.id2deprel) + 1
    config.embed_size = word_embeddings.shape[1]
    for (word_batch, tag_batch, deprel_batch), td_batch in \
            train_data.get_iterator(shuffled=False):
        config.n_word_features = word_batch.shape[-1]
        config.n_tag_features = tag_batch.shape[-1]
        config.n_deprel_features = deprel_batch.shape[-1]
        config.n_classes = td_batch.shape[-1]
        break
    print(f'# word features: {config.n_word_features}')
    print(f'# tag features: {config.n_tag_features}')
    print(f'# deprel features: {config.n_deprel_features}')
    print(f'# classes: {config.n_classes}')
    if debug:
        dev_sents = dev_sents[:500]
        dev_arcs = dev_arcs[:500]
        test_sents = test_sents[:500]
        test_arcs = test_arcs[:500]

    print(80 * '=')
    print('TRAINING')
    print(80 * '=')
    weights_file = Path('weights-q1.pt')
    print('Best weights will be saved to:', weights_file)
    model = ParserModel(transducer, config, word_embeddings)
    if torch.cuda.is_available():
        model = model.cuda()
    best_las = 0.
    trnbar_fmt = '{l_bar}{bar}| [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
    with tqdm(desc='Training', total=config.n_epochs, leave=False,
              unit='epoch', position=0, bar_format=trnbar_fmt) as progbar:
        for epoch in range(config.n_epochs):
            if debug:
                trn_loss = model.fit_epoch(list(islice(train_data, 32)), epoch,
                                           progbar, config.batch_size)
            else:
                trn_loss = model.fit_epoch(train_data, epoch, progbar)
            tqdm.write(f'Epoch {epoch + 1:>2} training loss: {trn_loss:.3g}')
            stdout.flush()
            dev_las, dev_uas = model.evaluate(dev_sents, dev_arcs)
            best = dev_las > best_las
            if best:
                best_las = dev_las
                if not debug:
                    torch.save(model.state_dict(), str(weights_file))
            tqdm.write(f'         validation LAS: {dev_las:.1%}'
                       f'{" (BEST!)" if best else  "        "} '
                       f'UAS: {dev_uas:.1%}')
    if not debug:
        print()
        print(80 * '=')
        print('TESTING')
        print(80 * '=')
        print('Restoring the best model weights found on the dev set.')
        model.load_state_dict(torch.load(str(weights_file)))
        stdout.flush()
        las, uas = model.evaluate(test_sents, test_arcs)
        if las:
            print(f'Test LAS: {las:.1%}', end='       ')
        print(f'UAS: {uas:.1%}')
        print('Done.')
    return 0


if __name__ == '__main__':
    main()
