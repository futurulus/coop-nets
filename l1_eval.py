#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Usage:
#   python l1_eval.py --run_dir runs/l2 --alpha 0.720
# l2 run dir should have s0_grids.0.jsons.gz and data.eval.jsons.
import gzip
import json
import numpy as np
from scipy.misc import logsumexp
import matplotlib.pyplot as plt
import os

from stanza.research import config, evaluate, metrics, output, instance

parser = config.get_options_parser()
parser.add_argument('--grids_file', help='Path to input grids.*.jsons.gz file.')
parser.add_argument('--alpha', type=float, default=0.720,
                    help='Inverse temperature parameter on the literal speaker.')


def evaluate_l1_eval():
    options = config.options(read=True)

    grids_path = os.path.join(options.run_dir, 's0_grids.0.jsons.gz')
    with gzip.open(grids_path, 'rb') as infile:
        grids = [json.loads(line.strip()) for line in infile]
    data_path = os.path.join(options.run_dir, 'data.eval.jsons')
    with open(data_path, 'r') as infile:
        insts = [instance.Instance(**json.loads(line.strip()))
                 for line in infile]

    assert len(grids) == len(insts), '{} != {}'.format(len(grids), len(insts))

    gold_outputs = np.array([inst.output for inst in insts])
    s0 = np.array([[np.array(ss['S0']).T for ss in grid['sets']]
                   for grid in grids])
    l1 = compute_l1(s0, alpha=options.alpha)
    l1_scores = l1[np.arange(l1.shape[0]), gold_outputs].tolist()
    l1_preds = np.argmax(l1, axis=1).tolist()

    m = [metrics.log_likelihood,
         metrics.log_likelihood_bits,
         metrics.perplexity,
         metrics.accuracy]
    learner = DummyLearner(l1_preds, l1_scores)

    results = evaluate.evaluate(learner, insts, metrics=m, split_id='l1_eval',
                                write_data=False)
    output.output_results(results, 'l1_eval')


def compute_l1(s0, alpha):
    unnorm = alpha * s0[:, 0, 0, :]
    return unnorm - logsumexp(unnorm, axis=1, keepdims=True)


class DummyLearner(object):
    def __init__(self, preds, scores):
        self.preds = preds
        self.scores = scores
        self.num_params = float('inf')

    def predict_and_score(self, *args, **kwargs):
        return self.preds, self.scores


if __name__ == '__main__':
    evaluate_l1_eval()
