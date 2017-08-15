import matplotlib.pyplot as plt
import seaborn as sns

############################################################################
# Helper functions for displaying medians and means
############################################################################

def show_stats(data, by, yvals, ax):
    medians = data.groupby([by])[yvals].median().values
    means = data.groupby([by])[yvals].mean().values
    stats_labels = ['median=%.2f\nmean=%.2f' % (medians[i], means[i])
                    for i in xrange(len(medians))]
    pos = range(len(medians))
    for tick,label in zip(pos, ax.get_xticklabels()):
        ax.text(pos[tick], medians[tick] + 0.5, stats_labels[tick],
                ha='center', size='x-small')

def show_vals(ax):
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2., height,
                '%.3f' % height, ha='center')

############################################################################
# Functions for box-/strip- and barplots
############################################################################

def boxplot(data, plot_file, ylabel, title='', strip=True):
    sns.set_style('white')
    if strip:
        ax = sns.stripplot(x='Language', y='Length', data=data, jitter=True,
                           marker='.', alpha=0.3)
    ax = sns.boxplot(x='Language', y='Length', data=data)
    show_stats(data, 'Language', 'Length', ax)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.savefig(plot_file, transparent=True, dpi=300)
    plt.gcf().clear()

def histogram(data, plot_file, ylabel, title=''):
    sns.set_style('ticks')
    fig, (ax_zh, ax_en) = plt.subplots(ncols=2, sharex=True, sharey=True)
    sns.despine()

    zh = data.loc[data['Language'] == 'Chinese']['Length'].tolist()
    en = data.loc[data['Language'] == 'English']['Length'].tolist()
    # zh = data.loc[data['Language'] == 'Chinese']['tokens'].tolist()
    # en = data.loc[data['Language'] == 'English']['tokens'].tolist()

    sns.distplot(zh, bins=15, ax=ax_zh, kde=False, norm_hist=True, hist_kws={'alpha':0.75}) # , rug=True)
    sns.distplot(en, bins=15, ax=ax_en, kde=False, norm_hist=True, hist_kws={'alpha':0.75}) # , rug=True)

    ax_zh.set_title('Chinese')
    ax_en.set_title('English')
    ax_zh.set_xlabel('Number of messages per round')
    ax_en.set_xlabel('Number of messages per round')
    ax_zh.set_ylabel('Proportion of rounds')
    # ax_zh.set_xlabel('Number of Jieba tokens')
    # ax_en.set_xlabel('Number of NLTK tokens')
    # ax_zh.set_ylabel('Proportion of messages')

    fig.suptitle(title)
    plt.savefig(plot_file, dpi=300)
    plt.gcf().clear()

def barplot(data, plot_file, ylabel, title=''):
    sns.set_style('ticks')
    ax = sns.barplot(x='Condition', y=data.columns.values[0], hue='Language',
                     order=['equal/far', 'further/split', 'closer/close'],
                     errcolor='#d1d1d1', data=data)
    sns.despine()
    show_vals(ax)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.savefig(plot_file, dpi=300) # transparent=True
    plt.legend(loc='upper left')
    plt.gcf().clear()
