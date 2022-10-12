#!/usr/bin/env python3
# Student name: NAME
# Student number: NUMBER
# UTORid: ID

from __future__ import annotations

import typing as T
from collections import OrderedDict
from math import inf, sqrt

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam

from transformers import AutoModel, AutoTokenizer
from transformers import logging as hftf_log

import config as cfg
from graphalg import is_single_root_tree, single_root_mst

BatchEncoding = T.Mapping[str, Tensor]
Batch = T.Tuple[BatchEncoding, Tensor, Tensor, Tensor, Tensor]


class GraphDepModel(nn.Module):
    def __init__(self, pretrained_model: str, n_deprels: int):
        super().__init__()
        hftf_log.set_verbosity_error()

        self.tknzr = AutoTokenizer.from_pretrained(pretrained_model,
                                                   cache_dir=cfg.MODEL_DIR)
        self.pt_model = AutoModel.from_pretrained(pretrained_model,
                                                  cache_dir=cfg.MODEL_DIR,
                                                  output_hidden_states=True)
        self.pt_model.requires_grad_(False)
        self.pt_width = self.pt_model.config.hidden_size
        self._pretrained_model = pretrained_model
        self._n_deprels = n_deprels

        self.create_arc_layers()
        self.create_label_layers()

        self.optim = Adam([p for p in self.parameters() if p.requires_grad],
                          lr=cfg.LR)

        self._register_state_dict_hook(self.__class__._on_save_state)

    def _on_save_state(self, state_dict: T.MutableMapping[str, T.Any],
                       prefix: str,
                       local_metadata: T.MutableMapping[str, T.Any]) -> None:
        local_metadata['pretrained_model'] = self._pretrained_model
        local_metadata['n_deprels'] = self._n_deprels
        for key in list(state_dict):
            if key.startswith('pt_model'):
                del state_dict[key]

    def load_state_dict(self, state_dict: OrderedDict[str, Tensor],
                        strict: bool = True):
        for k, v in state_dict._metadata[''].items():
            assert getattr(self, f'_{k}') == v
        incompatible = super().load_state_dict(state_dict, False)
        if strict:
            missing, unexpected = incompatible
            missing = [k for k in missing if not k.startswith('pt_model')]
            incompatible = incompatible._replace(missing_keys=missing)
            error_msgs = []
            if len(unexpected) > 0:
                error_msgs.insert(
                    0, 'Unexpected key(s) in state_dict: {}. '.format(
                        ', '.join('"{}"'.format(k) for k in unexpected)))
            if len(missing) > 0:
                error_msgs.insert(
                    0, 'Missing key(s) in state_dict: {}. '.format(
                        ', '.join('"{}"'.format(k) for k in missing)))

            if len(error_msgs) > 0:
                raise RuntimeError(
                    'Error(s) in loading state_dict for {}:\n\t{}'.format(
                        self.__class__.__name__, "\n\t".join(error_msgs)))
        return incompatible

    def create_arc_layers(self) -> None:
        """Creates layer weights and biases for arc scoring

        Our neural network computes predictions from the input vectors. The
        arc-scoring part of the model computes a transformation for each
        (candidate) head and a separate transformation for each (candidate)
        dependant. The outputs of these are then combined with a biaffine
        transformation.

        In this method, create the two transformation layers as well as the
        relevant tensors for the biaffine transformation.
        * Use nn.Sequential, nn.Linear, nn.ReLU, and nn.Dropout to create
          two separate transformations with the following properties:
            input size: self.pt_width
            hidden layer size: cfg.ARC_DIM
            output layer size: cfg.ARC_DIM
            activation functions for each layer: ReLU
            dropout probability (used after each activation): cfg.DROPOUT
          Assign the two transformations to self as attributes:
            self.arc_h_mlp
            self.arc_d_mlp
        * Create a 2D weight matrix for the biaffine transformation and assign
          it to self.arc_W. This weight matrix corresponds to W_A in the
          assignment handout; figure out what its dimensions should be based on
          how it is going to be used.
        * Create a vector for the head bias and assign it to self.arc_B. This
          corresponds to b_A in the assignment handout; again, figure out its
          dimensions based on how it is going to be used.
        * Make sure these last two Tensors are set as nn.Parameters.
        * Initialize self.arc_W and self.arc_B according to a uniform
          distribution on [-sqrt(3 / cfg.ARC_DIM), sqrt(3 / cfg.ARC_DIM)].

        Returns:
            None
        """
        # *#* BEGIN YOUR CODE *#* #
        # TODO: Do we need a second Linear for output layer?
        mlp = nn.Sequential(
            nn.Linear(self.pt_width, cfg.ARC_DIM),
            nn.Linear(cfg.ARC_DIM, cfg.ARC_DIM),
            nn.ReLU(),
            nn.Dropout(cfg.DROPOUT, True))
        self.arc_h_mlp = mlp
        self.arc_d_mlp = mlp
        self.arc_W = nn.Parameter(torch.randint(
            ((3 / cfg.ARC_DIM) ** 0.5) * -1, (3 / cfg.ARC_DIM) ** 0.5,
            (cfg.ARC_DIM, cfg.ARC_DIM)))
        self.arc_B = nn.Parameter(torch.randint(
            ((3 / cfg.ARC_DIM) ** 0.5) * -1, (3 / cfg.ARC_DIM) ** 0.5,
            cfg.ARC_DIM))
        # *** END YOUR CODE *** #

    def score_arcs(self, arc_head: Tensor, arc_dep: Tensor) -> Tensor:
        """
        Computes scores for candidate arcs

        arc_head and arc_dep represent the *transformed* inputs; the head- and
        dependant-specific transformations have already been applied for you
        (see self.forward()). This method computes the score using the
        biaffine weights defined earlier in the model, according to the
        specification in the assignment handout. Here, arc_head corresponds
        to H_A, arc_dep corresponds to D_A, self.arc_W corresponds to W_A,
        and self.arc_B corresponds to b_A.

        HINT: In order to implement this with the batched tensors that you
        receive in this method, you will likely find torch.einsum to be
        helpful.

        Args:
            arc_head (Tensor): Inputs (pre-)transformed to represent heads for
                candidate arcs. This Tensor has dtype float and dimensions
                (batch_sz, x, cfg.ARC_DIM).
            arc_dep (Tensor): Inputs (pre-)transformed to represent
                dependants for candidate arcs. This Tensor has dtype float and
                dimensions (batch_sz, y, cfg.ARC_DIM).

        Returns:
            Tensor of dtype float and dimensions (batch_sz, y, x) representing
            the score assigned to a candidate arc from x to y

            IMPORTANT: you will find that the values of x and y are equal,
            but it is important that you keep track of which is which and
            produce the correct ordering as specified above and in the
            assignment handout.
        """
        # *#* BEGIN YOUR CODE *#* #
        arc_scores = torch.einsum('bij, bjk -> bik', torch.matmul(arc_dep, self.arc_W), arc_head) +\
                     torch.matmul(arc_head, torch.unsqueeze(self.arc_B, 1))
        # *** END YOUR CODE *** #
        return arc_scores

    def create_label_layers(self) -> None:
        """
        Creates layer components for the label-scoring part of the model

        Our neural network computes predictions from the input vectors. As
        with the arc-scoring part, the label-scoring part of the model computes
        a transformation for each (candidate) head and a separate
        transformation for each (candidate) dependant. The outputs of these are
        then combined with a biaffine transformation. Unlike the arc-scoring
        part, here each candidate head-dependant pair has n_deprel possible
        classes, so the tensors here will be one order higher.

        In this method, create the two transformation layers as well as the
        relevant tensors for the biaffine transformation.
        * Use nn.Sequential, nn.Linear, nn.ReLU, and nn.Dropout to create
          two separate transformations with the following properties:
            input size: self.pt_width
            hidden layer size: cfg.LABEL_DIM
            output layer size: cfg.LABEL_DIM
            activation functions for each layer: ReLU
            dropout probability (used after each activation): cfg.DROPOUT
          Assign the two layers to self as attributes:
            self.label_h_mlp
            self.label_d_mlp
        * Create a 3D weight tensor for the biaffine transformation and assign
          it to self.label_W. This weight matrix corresponds to W_L in the
          assignment handout; figure out what its dimensions should be based on
          how it is going to be used.
        * Create a 2D weight matrix for the head-only score and assign it to
          self.label_h_W. This corresponds to W_Lh in the assignment handout;
          again, figure out its dimensions based on how it is going to be used.
        * Create a 2D weight matrix for the dependant-only score and assign
          it to self.label_d_W. This corresponds to W_Ld in the assignment
          handout; again, figure out its dimensions based on how it is going to
          be used.
        * Create a vector and assign for the label bias and assign it to
          self.label_B. This corresponds to b_L in the assignment handout;
          again, figure out its dimensions based on how it is going to be used.
        * Make sure these last four Tensors are set as nn.Parameters.
        * Initialize self.label_W, self.label_h_W, and self.label_d_w
          according to a uniform distribution on [-sqrt(3 / cfg.LABEL_DIM),
          sqrt(3 / cfg.LABEL_DIM)].
        * Initialize self.label_B to zeros.

        Returns:
            None
        """
        # *#* BEGIN YOUR CODE *#* #
        mlp = nn.Sequential(
            nn.Linear(self.pt_width, cfg.ARC_DIM),
            nn.Linear(cfg.ARC_DIM, cfg.ARC_DIM),
            nn.ReLU(),
            nn.Dropout(cfg.DROPOUT, True))
        self.label_h_mlp = mlp
        self.label_d_mlp = mlp
        self.label_W = nn.Parameter(torch.randint(
            ((3 / cfg.ARC_DIM) ** 0.5) * -1, (3 / cfg.ARC_DIM) ** 0.5,
            (cfg.ARC_DIM, 1, cfg.ARC_DIM)))
        self.label_h_W = nn.Parameter(torch.randint(
            ((3 / cfg.ARC_DIM) ** 0.5) * -1, (3 / cfg.ARC_DIM) ** 0.5,
            (cfg.ARC_DIM, cfg.ARC_DIM)))
        self.label_d_W = nn.Parameter(torch.randint(
            ((3 / cfg.ARC_DIM) ** 0.5) * -1, (3 / cfg.ARC_DIM) ** 0.5,
            (cfg.ARC_DIM, cfg.ARC_DIM)))
        self.label_B = nn.Parameter(torch.zeros(cfg.ARC_DIM))
        # *** END YOUR CODE *** #

    def score_labels(self, label_head: Tensor, label_dep: Tensor) -> Tensor:
        """
        Computes scores for candidate dependency relations for given arcs

        label_head and label_dep represent the *transformed* inputs; the head-
        and dependant-specific transformations have already been applied for
        you (see self.forward()). This method computes the score using the
        biaffine weights defined earlier in the model, according to the
        specification in the assignment handout. Here, label_head corresponds
        to H_L, label_dep corresponds to D_L, self.label_W corresponds to W_L,
        self.label_h_W corresponds to W_Lh, self.label_d_W corresponds to
        W_Ld, and self.label_B corresponds to b_L.

        HINT: In order to implement this with the batched tensors that you
        receive in this method, you will likely find torch.einsum to be
        helpful.

        Args:
            label_head (Tensor): Inputs (pre-)transformed to represent heads
                for candidate arc labels (dependency relations). This Tensor
                has dtype float and dimensions (batch_sz, x, cfg.LABEL_DIM).
            label_dep (Tensor): Inputs (pre-)transformed to represent
                dependants for candidate arc labels (dependency relations).
                This Tensor has dtype float and dimensions
                (batch_sz, y, cfg.LABEL_DIM).

        Returns:
            Tensor of dtype float and dimensions (batch_sz, y, x,
            self._n_deprels) representing the scores assigned to candidate
            dependency relations for an arc from x to y

            IMPORTANT: you will find that the values of x and y are equal,
            but it is important that you keep track of which is which and
            produce the correct ordering as specified above and in the
            assignment handout.
        """
        # *#* BEGIN YOUR CODE *#* #
        label_scores = torch.einsum('bij, bjk -> bik', torch.matmul(label_dep, self.arc_W), label_head) + \
                       torch.matmul(label_head, self.label_h_W) + \
                       torch.matmul(label_dep, self.label_d_W) + \
                       torch.unsqueeze(self.label_B, 1)
        # *** END YOUR CODE *** #
        return label_scores

    @property
    def device(self) -> torch.device:
        return next(self.pt_model.parameters()).device

    def transfer_batch(self, batch: Batch) -> Batch:
        pt_tok = {k: v.to(self.device) for k, v in batch[0].items()}
        orig_idx, heads, deprels, proj = (t.to(self.device) for t in batch[1:])
        return pt_tok, orig_idx, heads, deprels, proj

    def collate(self,
                batch: T.Iterable[T.Iterable[T.Iterable[str], T.Iterable[int],
                                             T.Iterable[int], bool]]) -> Batch:
        batch = list(zip(*batch))
        tok = self.tknzr(list(batch[0]), padding=True,
                         is_split_into_words=True, return_tensors='pt')
        orig_idx = torch.empty_like(tok['attention_mask'])
        for i, idxs in enumerate(orig_idx):
            for j, k in enumerate(tok.word_ids(i)[1:], start=1):
                if k is None:
                    idxs[j:] = idxs[j - 1]
                    break
                idxs[j] = k + 1
        orig_idx[:, 0] = 0
        return (tok, orig_idx[:, :-1], *[pad_sequences(s) for s in batch[1:3]],
                torch.tensor(batch[3]))

    def pt2orig_tok(self, pt_tok: BatchEncoding, orig_idx: Tensor) \
            -> T.Tuple[Tensor, Tensor]:
        # pre-trained model is frozen and no parameters here, so no need to
        # track gradients
        with torch.no_grad():
            lengths = orig_idx[:, -1]
            pt_out = self.pt_model(**pt_tok)['hidden_states']
            pt_out = torch.stack(pt_out).mean(0)
            orig_tok = torch.zeros(pt_out.shape[0], lengths.max() + 1,
                                   pt_out.shape[2], dtype=pt_out.dtype,
                                   device=pt_out.device)
            orig_idx = orig_idx.unsqueeze(-1)
            orig_tok.scatter_add_(1, orig_idx.expand(-1, -1, orig_tok.shape[2]),
                                  pt_out)
            counts = torch.zeros(*orig_tok.shape[:-1], 1, dtype=orig_idx.dtype,
                                 device=orig_idx.device)
            counts.scatter_add_(1, orig_idx, torch.ones_like(orig_idx))
            orig_tok /= counts.masked_fill_(counts == 0., 1.)
            return orig_tok, lengths

    def lengths2mask(self, lengths: Tensor) -> Tensor:
        lengths = lengths.unsqueeze(-1)
        range = torch.arange(lengths.max() + 1, device=lengths.device)
        mask = range <= lengths
        return mask.unsqueeze(-1) & mask.unsqueeze(-2)

    def _prepare_inputs(self, pt_tok: BatchEncoding, orig_idx: Tensor) \
            -> T.Tuple[Tensor, Tensor, Tensor]:
        inputs, lengths = self.pt2orig_tok(pt_tok, orig_idx)
        mask = self.lengths2mask(lengths)
        return inputs, lengths, mask

    def mask_possible(self, shape: T.Sequence[int]):
        """
        Creates a boolean mask that indicates which candidate dependencies are
        possible

        We can tell a priori that some arcs are disallowed. For example,
        regardless of the scores assigned to them, we know that loops (edges
        between a vertex and itself) are not valid dependencies. Similarly, we
        can infer a constraint on which arcs can be assigned the root
        dependency relation (label).

        Implement this function so that it returns a boolean tensor of the
        given shape, where each entry indicates whether that position in the
        tensor corresponds to an allowable dependency (both arc and label).
        Remember that all you know is the candidate head, dependant, and
        dependency relation, so none of the contraints you represent here
        involve how edges might interact. You also don't know anything about
        the sentence lengths here, so you aren't to try to mask out padding
        values (that is already done for you in the lengths2mask method).
        The starter code creates a tensor of the relevant shape that allows
        all possibilities: you must set the disallowed entries to False.

        In your report, include a brief writeup (~1 paragraph per constraint)
        that explains which constraints you enforce and how you do so.

        Args:
            shape (Sequence[int]): the shape for this batch. There are four
                dimensions: (batch_sz, y, x, self._n_deprels)

        Returns:
            A boolean Tensor, where each element is True if and only if a
            dependency corresponding to that element is allowable. In other
            words, element at index (b, i, j, k), ask whether it is possible
            for there to be a dependency from vertex j to vertex i having
            dependency relation k, based solely on the values of i, j, and k.
            (The value for the first dimension indexes the batch dimension, so
            is irrelevant to this.)

            IMPORTANT:
            * The ROOT vertex is at position 0; i.e., the element at index
              (b, i, 0, k) indicates a dependency *from* ROOT to vertex i with
              dependency relation k.
            * The root dependency relation has value 0; i.e., the element at
              index (b, i, j, 0) indicates a dependency from vertex j to vertex
              i with dependency relation (i.e., label) root.

            HINT: You don't need to know the indices for the other dependency
            relations; the text above tells you everything you need to know,
            and this question isn't worth very many marks, so don't overthink
            it!
        """
        mask = torch.ones(shape, dtype=torch.bool, device=self.device)
        # *#* BEGIN YOUR CODE *#* #
        # *** END YOUR CODE *** #
        return mask

    def forward(self, inputs: Tensor) -> T.Tuple[Tensor, Tensor]:
        arc_scores = self.score_arcs(self.arc_h_mlp(inputs),
                                     self.arc_d_mlp(inputs))
        label_scores = self.score_labels(self.label_h_mlp(inputs),
                                         self.label_d_mlp(inputs))
        mask = self.mask_possible(label_scores.shape)
        arc_scores = arc_scores.masked_fill_(~mask.any(-1), -inf)
        label_scores = label_scores.masked_fill_(~mask, -inf)

        return arc_scores, label_scores

    def loss(self, arc_scores: Tensor, label_scores: Tensor, heads: Tensor,
             deprels: Tensor, mask: Tensor) -> Tensor:
        arc_scores = arc_scores[:, 1:].reshape(-1, arc_scores.shape[-1])
        heads_ = heads.view(*heads.shape, 1, 1).expand(-1, -1, -1,
                                                       label_scores.shape[-1])
        label_scores = label_scores[:, 1:].gather(2, heads_)
        label_scores = label_scores.view(-1, label_scores.shape[-1])
        heads = heads.view(arc_scores.shape[0])
        deprels = deprels.view_as(heads)
        xent = (F.cross_entropy(arc_scores, heads, reduction='none')
                + F.cross_entropy(label_scores, deprels, reduction='none'))
        mask = mask[:, 1:].any(-1).view_as(heads)
        xent = xent * mask
        return xent.sum() / mask.sum()

    def train_batch(self, batch: Batch) -> float:
        self.train()
        pt_tok, orig_idx, heads, deprels, proj = self.transfer_batch(batch)
        self.optim.zero_grad()
        inputs, _, mask = self._prepare_inputs(pt_tok, orig_idx)
        arc_scores, label_scores = self(inputs)
        loss = self.loss(arc_scores, label_scores, heads, deprels, mask)
        loss.backward()
        self.optim.step()
        return loss.item()

    def _predict_batch(self, batch: Batch) \
            -> T.Tuple[Tensor, Tensor, Tensor, Tensor]:
        pt_tok, orig_idx, _, _, _ = batch
        inputs, lengths, mask = self._prepare_inputs(pt_tok, orig_idx)
        arc_scores, label_scores = self(inputs)
        arc_scores = arc_scores.masked_fill_(~mask, -inf)
        label_scores = label_scores.masked_fill_(~mask.unsqueeze(-1), -inf)
        best_arcs = single_root_mst(arc_scores, lengths)[:, 1:].unsqueeze(-1)
        best_labels = label_scores[:, 1:].argmax(-1).gather(2, best_arcs)
        return (best_arcs.squeeze(-1), best_labels.squeeze(-1),
                mask[:, 1:].any(-1), lengths)

    def eval_batch(self, batch: Batch) -> T.Tuple[int, int, int, int]:
        self.eval()
        _, _, heads, deprels, _ = batch = self.transfer_batch(batch)
        pred_heads, pred_labels, mask, lengths = self._predict_batch(batch)
        arcs_acc = (pred_heads == heads) & mask
        label_acc = (arcs_acc & (pred_labels == deprels)) & mask
        ret = (arcs_acc, label_acc, mask, is_single_root_tree(pred_heads,
                                                              lengths))
        return tuple(t.sum().item() for t in ret)


def pad_sequences(seqs: T.Iterable[T.Iterable[T.Union[bool, int, float]]]) \
        -> Tensor:
    return pad_sequence([torch.tensor(s) for s in seqs], True)
