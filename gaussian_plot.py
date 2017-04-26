# coding: utf-8
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
    visualize_integrated(score_fn, 'drab green not the bluer one', aspect=1.5,
                         special_points=special_points,
                         save='writing/2016/figures/gaussian.pdf')


def get_score_fn(model):
    def score_fn(description, colors):
        mean, covar = model.get_gaussian_params(description)
        colors_vec = model.color_vec.vectorize_all(colors)
        diff = colors_vec - mean

        BATCH_SIZE = 256
        scores = []
        for start in range(0, colors_vec.shape[0], BATCH_SIZE):
            dbatch = diff[start:start + BATCH_SIZE]
            scores_batch = (dbatch.dot(covar) * dbatch).sum(axis=1)
            scores.extend(scores_batch.tolist())
        return scores

    return score_fn


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
    figwidth =  (((100 + 360. / aspect) / 10.) + 1) / 3.
    figheight = ((100 / 10.) + 2) / 3.
    gs = gridspec.GridSpec(3, 3,
                           width_ratios=[1, 10, 36. / aspect],
                           height_ratios=[10, 1, 1],
                           wspace=0.2 * (figheight / figwidth),
                           hspace=0.12)
    fig = plt.figure(figsize=(figwidth, figheight))

    vax, sv, hv, ignored, sax, hax = [plt.subplot(gs[i]) for i in range(6)]
    sv.xaxis.set_visible(False)
    sv.yaxis.set_visible(False)
    hv.xaxis.set_visible(False)
    hv.yaxis.set_visible(False)
    vax.xaxis.set_visible(False)
    sax.yaxis.set_visible(False)
    hax.yaxis.set_visible(False)
    ignored.axis('off')

    return fig, (hv, sv, hax, sax, vax)


def visualize_integrated(score_fn, description, aspect=1., interpolate=True,
                         special_points=None, save=None):
    if special_points is None:
        special_points = {}
    hv, hs, sv = get_scores_grid(score_fn, description)

    interp = None if interpolate else 'none'
    fig, (hvax, svax, hax, sax, vax) = integrated_subplots(aspect=aspect)

    cross_size = 40.0
    text_size = 14
    shadow_offset = (0.75, -0.5)
    text_offset = (-17.5, 2.5)

    for k, (h, s, v) in special_points.items():
        rgb = hsv_to_rgb(h / 360.0, s / 100.0, v / 100.0)

        hvax.scatter([h + shadow_offset[0]], [v + shadow_offset[1]],
                     marker='+', s=cross_size, c='black')
        hvax.scatter([h], [v], marker='+', s=cross_size, c=rgb)
        hvax.annotate(k, xy=(h, v), color='black', size=text_size,
                      xytext=(h + text_offset[0] * aspect + shadow_offset[0],
                              v + text_offset[1] * aspect + shadow_offset[1]))
        hvax.annotate(k, xy=(h, v), color=rgb, size=text_size,
                      xytext=(h + text_offset[0] * aspect,
                              v + text_offset[1] * aspect))

        svax.scatter([s + shadow_offset[0]], [v + shadow_offset[1]],
                     marker='+', s=cross_size, c='black')
        svax.scatter([s], [v], marker='+', s=cross_size, c=rgb)
        svax.annotate(k, xy=(s, v), color='black', size=text_size,
                      xytext=(s + text_offset[0] + shadow_offset[0],
                              v + text_offset[1] + shadow_offset[1]))
        svax.annotate(k, xy=(s, v), color=rgb, size=text_size,
                      xytext=(s + text_offset[0],
                              v + text_offset[1]))

    hvax.imshow(hv, cmap='gray', interpolation=interp,
                aspect=aspect, extent=[0, 360, 0, 100])
    svax.imshow(sv, cmap='gray', interpolation=interp, extent=[0, 100, 0, 100])

    gradient_h = np.arange(0, 1, 1. / hv.shape[0])[np.newaxis, :]
    print(gradient_h.shape)
    gradient_s = np.arange(0, 1, 1. / sv.shape[0])[np.newaxis, :]
    print(gradient_s.shape)
    gradient_v = np.arange(1, 0, -1. / sv.shape[1])[:, np.newaxis]
    print(gradient_v.shape)

    hax.imshow(gradient_h, cmap=COLORMAPS['h'],
               aspect=aspect, extent=[0, 360, 0, 5])
    hax.set_xlabel('Hue', fontsize=18)
    hax.set_xticks(np.arange(0, 360, 60))
    sax.imshow(gradient_s, cmap=COLORMAPS['s'], extent=[0, 100, 0, 5])
    sax.set_xlabel('Saturation', fontsize=18)
    vax.imshow(gradient_v, cmap=COLORMAPS['v'], extent=[0, 5, 0, 100])
    vax.set_ylabel('Value', fontsize=18)
    # plt.suptitle('"%s"' % description, fontsize=24)
    if save is not None:
        plt.savefig(os.path.expanduser(save), format='pdf')
    plt.show()


if __name__ == '__main__':
    make_gaussian_plot()
