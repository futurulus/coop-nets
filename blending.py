#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Usage:
#   python blending.py --run_dir runs/l2 --blend_name additive \
#                      --base_weight 0.099 \
#                      --speaker_weight 0.297 \
#                      --alpha 0.555 \
#                      --alpha_l1 1.293 \
#                      --additive true
#   python blending.py --run_dir runs/l2 --blend_name fullblend \
#                      --base_weight -0.15 \
#                      --speaker_weight 0.608 \
#                      --alpha 0.544 \
#                      --gamma 0.509 \
#                      --additive false
# l2 run dir should have s0_grids.0.jsons.gz and data.eval.jsons.
import gzip
import json
import numpy as np
from scipy.misc import logsumexp
import matplotlib.pyplot as plt
import os

from stanza.research import config, evaluate, metrics, output, instance

parser = config.get_options_parser()
parser.add_argument('--blend_name', default='blend',
                    help='Name of blending strategy, to prepend to eval keys and filenames.')
parser.add_argument('--base_weight', type=float, default=0.099,
                    help='Weight of L0 model (blending L0 and L2 in Lb).')
parser.add_argument('--speaker_weight', type=float, default=0.297,
                    help='Weight of L1 model (blending L0 and L1 in La).')
parser.add_argument('--alpha', type=float, default=0.555,
                    help='Rationality of S1 model (inverse temperature).')
parser.add_argument('--alpha_l1', type=float, default=1.293,
                    help='Rationality of L1 model (inverse temperature).')
parser.add_argument('--gamma', type=float, default=0.509,
                    help='Weight of Lb model (blending La and Lb).')
parser.add_argument('--additive', type=config.boolean, default=True,
                    help='If True, average probabilities instead of log probabilities.')


def evaluate_ak_blending():
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
    l0 = np.array([[np.array(ss['L0']).T for ss in grid['sets']]
                   for grid in grids])
    s0 = np.array([[np.array(ss['S0']).T for ss in grid['sets']]
                   for grid in grids])
    if options.additive:
        ak = compute_additive(l0, s0,
                              bw=options.base_weight,
                              sw=options.speaker_weight,
                              alpha_s1=options.alpha,
                              alpha_l1=options.alpha_l1)
    else:
        ak = compute_ak(l0, s0,
                        bw=options.base_weight,
                        sw=options.speaker_weight,
                        alpha=options.alpha,
                        gamma=options.gamma)
    ak_scores = ak[np.arange(ak.shape[0]), gold_outputs].tolist()
    ak_preds = np.argmax(ak, axis=1).tolist()

    m = [metrics.log_likelihood,
         metrics.log_likelihood_bits,
         metrics.perplexity,
         metrics.accuracy]
    learner = DummyLearner(ak_preds, ak_scores, params={
        'base_weight': options.base_weight,
        'speaker_weight': options.speaker_weight,
        'alpha': options.alpha,
        'alpha_l1': options.alpha_l1,
        'gamma': options.gamma,
        'additive': options.additive,
    })

    split_id = '{}_eval'.format(options.blend_name)
    results = evaluate.evaluate(learner, insts, metrics=m,
                                split_id=split_id,
                                write_data=False)

    output.output_results(results, split_id)

    options_dump = vars(options)
    del options_dump['overwrite']
    del options_dump['config']
    config.dump_pretty(options_dump, split_id + '_config.json')


def compute_ak(l0, s0, bw, sw, alpha, gamma):
    l0 = normalize(np.maximum(l0, -1000.0), axis=3)

    ak = normalize((1 - sw) * l0[:, 0, 0, :] + sw * s0[:, 0, 0, :])

    s1 = normalize(l0 * alpha, axis=2)
    l2 = normalize(s1, axis=3)
    lstar_ss = normalize(bw * l0[:, :, 0, :] + (1 - bw) * l2[:, :, 0, :], axis=2)
    lstar = normalize(log_average(lstar_ss))

    return normalize((1 - gamma) * ak + gamma * lstar)


def compute_additive(l0, s0, bw, sw, alpha_s1, alpha_l1):
    s1 = normalize(l0 * alpha_s1, axis=2)
    l2_ss = normalize(s1[:, :, 0, :], axis=2)
    l2 = normalize(log_average(l2_ss))

    l1 = normalize(s0[:, 0, 0, :] * alpha_l1)

    l0 = l0[:, 0, 0, :]

    return log_weighted_ave([l0, l1, l2], [bw, sw, 1.0 - bw - sw])


def normalize(arr, axis=1, cap=-100.0):
    arr = np.maximum(arr, cap)
    return arr - logsumexp(arr, axis=axis, keepdims=True)


def log_average(arr, axis=1):
    return logsumexp(arr, axis=axis) - np.log(arr.shape[axis])


def log_weighted_ave(arrs, weights):
    arrs = np.array(arrs)
    log_weights = np.log(weights)
    log_weights -= logsumexp(log_weights)
    for _ in range(len(arrs.shape) - 1):
        log_weights = log_weights[..., np.newaxis]
    return logsumexp(arrs + log_weights, axis=0)


class DummyLearner(object):
    def __init__(self, preds, scores, params):
        self.preds = preds
        self.scores = scores
        self.num_params = params

    def predict_and_score(self, *args, **kwargs):
        return self.preds, self.scores


if __name__ == '__main__':
    evaluate_ak_blending()
