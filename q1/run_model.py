#!/usr/bin/env python3

if __name__ == '__main__':
    import typing as T
    from operator import itemgetter
    from pathlib import Path

    import torch

    from conllu import TokenList

    from data import load_and_preprocess_data
    from model import Config, ParserModel
    from parse import minibatch_parse

    torch.manual_seed(1234)
    if gpu := torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        torch.cuda.manual_seed_all(1234)
        print(f'Running on GPU: {torch.cuda.get_device_name()}.')
    else:
        print('Running on CPU.')
    config = Config()
    data = load_and_preprocess_data(max_batch_size=config.batch_size,
                                    transition_cache=0)
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

    weights_file = Path('weights-q1.pt')
    print(f'Loading model weights from {weights_file}...', end=' ', flush=True)
    model = ParserModel(transducer, config, word_embeddings)
    model.load_state_dict(torch.load(str(weights_file),
                                     map_location='cuda' if gpu else 'cpu'))
    if gpu:
        model = model.cuda()
    model.eval()
    print('Done.', flush=True)
    dev_out, tst_out = Path('dev.out.conllu'), Path('tst.out.conllu')
    print(f'Outputs will be saved to {dev_out} and {tst_out}.\n')

    def run_and_save(sentences: T.Iterable[T.Iterable[T.Tuple[str, str]]],
                     filepath: Path, desc: str):
        print(f'Running model on {desc.lower()}...')
        pred_deps = minibatch_parse(sentences, model, model.config.batch_size)
        pred_deps = [sorted(preds, key=itemgetter(1)) for preds in pred_deps]
        with filepath.open('w') as fout:
            for sent, pdeps in zip(sentences, pred_deps):
                if len(sent) == len(pdeps):
                    tl = TokenList(
                        [{'id': i, 'form': w, 'lemma': '_', 'upos': '_',
                          'xpos': '_', 'feats': '_', 'head': h, 'deprel': l,
                          'deps': '_', 'misc': '_'}
                         for (w, p), (h, i, l) in zip(sent, pdeps)])
                else:
                    tl = TokenList([{'id': i, 'form': w, 'lemma': '_',
                                     'upos': '_', 'xpos': '_', 'feats': '_',
                                     'head': '_', 'deprel': '_', 'deps': '_',
                                     'misc': '_'}
                                    for i, (w, p) in enumerate(sent, start=1)])
                    for h, i, l in pdeps:
                        i -= 1
                        tl[i]['head'], tl[i]['deprel'] = h, l
                fout.write(tl.serialize())

    run_and_save(dev_sents, dev_out, 'Dev set')
    run_and_save(test_sents, tst_out, 'Test set')
