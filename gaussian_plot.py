# coding: utf-8
import json
from scipy.misc import logsumexp
import numpy as np
import os
import cPickle as pickle
import matplotlib.pyplot as plt
plt.rc('font', family='serif')
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from colorsys import hsv_to_rgb

from stanza.research import instance

import run_experiment

SAT_CMAP = {
    'red':   ((0.0, 0.5, 0.5),
              (1.0, 1.0, 1.0)),
    'green': ((0.0, 0.5, 0.5),
              (1.0, 0.0, 0.0)),
    'blue':  ((0.0, 0.5, 0.5),
              (1.0, 0.0, 0.0)),
}

VAL_CMAP = {
    'red':   ((0.0, 0.0, 0.0),
              (1.0, 1.0, 1.0)),
    'green': ((0.0, 0.0, 0.0),
              (1.0, 0.0, 0.0)),
    'blue':  ((0.0, 0.0, 0.0),
              (1.0, 0.0, 0.0)),
}

COLORMAPS = {
    'h': 'hsv',
    's': LinearSegmentedColormap('sat', SAT_CMAP),
    'v': LinearSegmentedColormap('val', VAL_CMAP),
}


def make_gaussian_plot():
    with open('runs/l0_gaussian/quickpickle.p', 'rb') as infile:
        model = pickle.load(infile)

    score_fn = get_score_fn(model)
    special_points = {
        'target': (155.0, 20.4, 55.7),
        'distractor 1': (193.0, 37.6, 61.6),
        'distractor 2': (72.0, 71.2, 77.6),
    }

    print_scores(score_fn, 'drab green not the bluer one', special_points)
    visualize_integrated(score_fn, 'drab green not the bluer one', aspect=1.5,
                         special_points=special_points,
                         save='writing/2016/figures/gaussian.pdf')


def print_scores(score_fn, description, context):
    scores_vec = []
    scores_map = {}
    for name, c in context.iteritems():
        score = score_fn(description, [c])[0]
        scores_vec.append(score)
        scores_map[name] = score

    for name in sorted(scores_map.keys()):
        print('{}: p = {}'.format(name, scores_map[name]))


def get_score_fn(model):
    def score_fn(description, colors):
        # mean, covar = model.get_gaussian_params(description)
        mean, covar = load_gaussian_params()
        colors_vec = model.color_vec.vectorize_all(colors, hsv=True)
        print('points: {}'.format(colors_vec.round(3)))
        diff = colors_vec - mean

        BATCH_SIZE = 256
        scores = []
        for start in range(0, colors_vec.shape[0], BATCH_SIZE):
            dbatch = diff[start:start + BATCH_SIZE]
            scores_batch = (dbatch.dot(covar) * dbatch).sum(axis=1)
            scores.extend(scores_batch.tolist())
        return scores

    return score_fn


def load_gaussian_params():
    with open('runs/l0_gaussian/params.json', 'r') as infile:
        params = json.loads(infile.read().strip())
    return np.array(params['mean']), np.array(params['covar'])


def get_scores_grid(score_fn, description):
    colors = [
        (h, s, v)
        for h in range(2, 360, 4)
        for s in range(2, 100, 4)
        for v in range(2, 100, 4)
    ]
    scores_grid = np.array(score_fn(description, colors)).reshape((90, 25, 25))

    probs = np.exp(scores_grid)
    sums = [probs.sum(axis=a) for a in (1, 2, 0)]
    for i in range(len(sums)):
        sums[i] /= sums[i].sum()
        sums[i] = np.log(sums[i]).transpose()[::-1, :]

    return sums


def integrated_subplots(aspect=1.):
    figwidth =  (((360. / aspect) / 10.) + 1) / 3.
    figheight = ((100 / 10.) + 2) / 3.
    gs = gridspec.GridSpec(3, 2,
                           width_ratios=[1, 36. / aspect],
                           height_ratios=[10, 1, 1],
                           wspace=0.2 * (figheight / figwidth),
                           hspace=0.12)
    fig = plt.figure(figsize=(figwidth, figheight))

    sax, hs, ignored, hax = [plt.subplot(gs[i]) for i in range(4)]
    hs.xaxis.set_visible(False)
    hs.yaxis.set_visible(False)
    sax.xaxis.set_visible(False)
    hax.yaxis.set_visible(False)
    ignored.axis('off')

    return fig, (hs, hax, sax)


def visualize_integrated(score_fn, description, aspect=1., interpolate=True,
                         special_points=None, save=None):
    if special_points is None:
        special_points = {}
    hv, hs, sv = get_scores_grid(score_fn, description)

    interp = None if interpolate else 'none'
    fig, (hsax, hax, sax) = integrated_subplots(aspect=aspect)

    cross_size = 40.0
    text_size = 14
    shadow_offset = (0.75, -0.5)
    text_offset = (-17.5, 2.5)

    for k, (h, s, v) in special_points.items():
        rgb = hsv_to_rgb(h / 360.0, s / 100.0, v / 100.0)

        hsax.scatter([h + shadow_offset[0]], [s + shadow_offset[1]],
                     marker='+', s=cross_size, c='black')
        hsax.scatter([h], [s], marker='+', s=cross_size, c=rgb)
        ann_h = h - 5. * (k == 'target')
        ann_s = 100. + 5. * (k == 'target')
        arrowprops={'edgecolor': 'black',
                    'arrowstyle': '->',
                    'relpos': (0.5, 0)}
        hsax.annotate(k, xy=(h + shadow_offset[0],
                             s + shadow_offset[1] + 3.0), color='black', size=text_size,
                      arrowprops=dict(arrowprops),
                      xytext=(ann_h + text_offset[0] * aspect + shadow_offset[0],
                              ann_s + text_offset[1] * aspect + shadow_offset[1]))
        arrowprops['edgecolor'] = rgb
        hsax.annotate(k, xy=(h, s + 3.0), color=rgb, size=text_size,
                      arrowprops=dict(arrowprops),
                      xytext=(ann_h + text_offset[0] * aspect,
                              ann_s + text_offset[1] * aspect))

    hsax.imshow(hs, cmap='gray', interpolation=interp,
                aspect=aspect, extent=[0, 360, 0, 100])

    gradient_h = np.arange(0, 1, 1. / hs.shape[0])[np.newaxis, :]
    print(gradient_h.shape)
    gradient_s = np.arange(1, 0, -1. / hs.shape[1])[:, np.newaxis]
    print(gradient_s.shape)

    hax.imshow(gradient_h, cmap=COLORMAPS['h'],
               aspect=aspect, extent=[0, 360, 0, 5])
    hax.set_xlabel('Hue', fontsize=18)
    hax.set_xticks(np.arange(0, 360, 60))
    sax.imshow(gradient_s, cmap=COLORMAPS['s'], extent=[0, 5, 0, 100])
    sax.set_ylabel('Saturation', fontsize=18)
    # plt.suptitle('"%s"' % description, fontsize=24)
    if save is not None:
        plt.savefig(os.path.expanduser(save), format='pdf')
    plt.show()


if __name__ == '__main__':
    make_gaussian_plot()
