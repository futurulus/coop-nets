import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

ZH_COLOR = '#EE3224'
EN_COLOR = '#F78F1E'

# def boxplot(data, plot_file, xticks=[], xlabel='', ylabel='', title=''):
    # fig = plt.figure(1, figsize=(8,6))
    # ax = fig.add_subplot(111)
    # bp = ax.boxplot(data, patch_artist=True, showmeans=True)
    # bp['boxes'][0].set(facecolor=ZH_COLOR, alpha=0.5)
    # bp['boxes'][1].set(facecolor=EN_COLOR, alpha=0.5)
    # for median in bp['medians']:
    #     median.set(color='#75e3ff')
    # for mean in bp['means']:
    #     mean.set(marker='.')
    # for flier in bp['fliers']:
    #     flier.set(marker='.', color='#e7298a', alpha=0.5)
    # for i in xrange(len(data)):
    #     d = data[i]
    #     x, y = bp['medians'][i].get_xydata()[1]
    #     ax.annotate('median=%.3f,\nmean=%.3f' % (np.median(d), np.mean(d)),
    #                 (x,y), (x+0.05,1))
    # ax.set_xticklabels(xticks)
    # ax.set_xlabel(xlabel)
    # ax.set_ylabel(ylabel)
    # ax.set_title(title)
    # fig.savefig(plot_file, bbox_inches='tight', dpi=300)
    # plt.gcf().clear()

def stripplot(data, plot_file, xticks=[], xlabel='', ylabel='', title=''):
    sns.set_style('white')
    ax = sns.stripplot(data=data, jitter=True, marker='.', alpha=0.1)
    ax = sns.boxplot(data=data)
    ax.set_xticklabels(xticks)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.savefig(plot_file, dpi=300)
    plt.gcf().clear()

def barplot(data):
    sns.set_style('white')
    ax = sns.barplot(x='condition', y='usages', hue='language', data=data)
    plt.savefig('TEST.png', dpi=300)
    plt.gcf().clear()

def show_values(ax, rects):
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height+0.0005,
                '%.3f' % height, ha='center', va='bottom')

def bargraph(data, plot_file, xticks=[], series_names=[], xlabel='', ylabel='', title=''):
    pos = np.arange(len(data[0]))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8,6))
    zh_rects = ax.bar(pos, data[0], width, alpha=0.5, color=ZH_COLOR)
    en_rects = ax.bar(pos + width, data[1], width, alpha=0.5, color=EN_COLOR)
    show_values(ax, zh_rects)
    show_values(ax, en_rects)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks([p + 0.5 * width for p in pos])
    ax.set_xticklabels(xticks)
    plt.legend(series_names, loc='upper left')
    fig.savefig(plot_file, dpi=300)
