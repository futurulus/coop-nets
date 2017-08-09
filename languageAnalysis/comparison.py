#coding:utf-8
import matplotlib.pyplot as plt
import numpy as np
import enchant
import csv

ZH_MSG_FILE = '../hawkins_data/colorReferenceMessageChinese.csv'
ZH_CLICK_FILE = '../hawkins_data/colorReferenceClicksChinese.csv'
EN_MSG_FILE = '../behavioralAnalysis/humanOutput/filteredCorpus.csv'
PUNCTUATION = ['~',',','.','?','!','。','，','？','！','”']

def msg_lengths(lang='english'):
    '''
    Returns the average length of the messages sent for a given language
    (English or Chinese). For English, it counts the number of words
    (after removing punctuation and splitting on whitespaces), and for Chinese,
    it counts the number of chars (after removing punctuation, whitespaces,
    and any messages containing English).
    '''
    msg_file = ZH_MSG_FILE if lang == 'chinese' else EN_MSG_FILE
    msg_lengths = []
    with open(msg_file, 'r') as f:
        for row in csv.DictReader(f):
            msg = row['contents']
            for c in PUNCTUATION:
                msg = msg.replace(c, '')
            if lang == 'chinese':
                d = enchant.Dict('en_US')
                if any([d.check(s) for s in msg.split()]):
                    msg = ''
                else:
                    msg = msg.replace(' ', '')
                msg_lengths.append(len(msg.decode('utf8')))
            else:
                msg_lengths.append(len(msg.split()))
    return msg_lengths

def dlg_lengths(lang='english'):
    '''
    Returns the average number of messages exchanged for a single round
    for a given language (English or Chinese).
    '''
    msg_file = ZH_MSG_FILE if lang == 'chinese' else EN_MSG_FILE
    dlg_lengths = []
    with open(msg_file, 'r') as f:
        counts, gameid = {}, None
        for row in csv.DictReader(f):
            if gameid == row['gameid']:
                try:
                    counts[row['roundNum']] += 1
                except KeyError:
                    counts[row['roundNum']] = 1
            else:
                avg_round_length = np.mean(counts.values()) if counts.values() else 0
                dlg_lengths.append(avg_round_length)
                counts, gameid = {}, row['gameid']
    return dlg_lengths

def superlative_usage(lang='english'):
    '''
    Returns how often superlatives are used on each of the three conditions
    (far, split, close) for a given language (English or Chinese).
    '''
    msg_file = ZH_MSG_FILE if lang == 'chinese' else EN_MSG_FILE
    click_file = ZH_CLICK_FILE if lang == 'chinese' else EN_MSG_FILE
    if lang == 'chinese':
        sups = {'further': [], 'closer': [], 'equal' : []}
        click_rows, msg_rows = [], []
        with open(click_file, 'r') as click_f:
            for click_row in csv.DictReader(click_f):
                click_rows.append(click_row)
        with open(msg_file, 'r') as msg_f:
            for msg_row in csv.DictReader(msg_f):
                msg_rows.append(msg_row)
        for click_row in click_rows:
            cond = click_row['condition']
            gameid = click_row['gameid']
            game_msgs = [x['contents'] for x in msg_rows if x['gameid'] == gameid]
            for msg in game_msgs:
                if '最' in msg:
                    sups[cond].append(1)
                else:
                    sups[cond].append(0)
        return np.mean(sups['further']), np.mean(sups['closer']), np.mean(sups['equal'])
    else:
        sups = {'far': [], 'split': [], 'close' : []}
        with open(click_file, 'r') as f:
            for row in csv.DictReader(f):
                cond = row['condition']
                if 'est' in row['contents'] or ('most' in row['contents']
                                            and 'almost' not in row['contents']):
                    sups[cond].append(1)
                else:
                    sups[cond].append(0)
        return np.mean(sups['far']), np.mean(sups['split']), np.mean(sups['close'])

def comparative_usage(lang='english'):
    '''
    Returns how often comparatives are used on each of the three conditions
    (far, split, close) for a given language (English or Chinese).
    '''
    msg_file = ZH_MSG_FILE if lang == 'chinese' else EN_MSG_FILE
    click_file = ZH_CLICK_FILE if lang == 'chinese' else EN_MSG_FILE
    if lang == 'chinese':
        comps = {'further': [], 'closer': [], 'equal' : []}
        click_rows, msg_rows = [], []
        with open(click_file, 'r') as click_f:
            for click_row in csv.DictReader(click_f):
                click_rows.append(click_row)
        with open(msg_file, 'r') as msg_f:
            for msg_row in csv.DictReader(msg_f):
                msg_rows.append(msg_row)
        for click_row in click_rows:
            cond = click_row['condition']
            gameid = click_row['gameid']
            game_msgs = [x['contents'] for x in msg_rows if x['gameid'] == gameid]
            for msg in game_msgs:
                if '更' in msg or '多' in msg or '少' in msg: # '比较' in msg
                    comps[cond].append(1)
                else:
                    comps[cond].append(0)
        return np.mean(comps['further']), np.mean(comps['closer']), np.mean(comps['equal'])
    else:
        comps = {'far': [], 'split': [], 'close' : []}
        with open(click_file, 'r') as f:
            for row in csv.DictReader(f):
                cond = row['condition']
                if ('er ' in row['contents'] and 'other ' not in row['contents'] \
                                            and 'water' not in row['contents'] \
                                            and 'hunter' not in row['contents']) \
                    or 'more' in row['contents'] or 'less ' in row['contents']:
                    comps[cond].append(1)
                else:
                    comps[cond].append(0)
    return np.mean(comps['far']), np.mean(comps['split']), np.mean(comps['close'])

def negation_usage(lang='english'):
    '''
    Returns how often negations are used on each of the three conditions
    (far, split, close) for a given language (English or Chinese).
    '''
    msg_file = ZH_MSG_FILE if lang == 'chinese' else EN_MSG_FILE
    click_file = ZH_CLICK_FILE if lang == 'chinese' else EN_MSG_FILE
    if lang == 'chinese':
        negs = {'further': [], 'closer': [], 'equal' : []}
        click_rows, msg_rows = [], []
        with open(click_file, 'r') as click_f:
            for click_row in csv.DictReader(click_f):
                click_rows.append(click_row)
        with open(msg_file, 'r') as msg_f:
            for msg_row in csv.DictReader(msg_f):
                msg_rows.append(msg_row)
        for click_row in click_rows:
            cond = click_row['condition']
            gameid = click_row['gameid']
            game_msgs = [x['contents'] for x in msg_rows if x['gameid'] == gameid]
            for msg in game_msgs:
                if '不' in msg:
                    negs[cond].append(1)
                else:
                    negs[cond].append(0)
        return np.mean(negs['further']), np.mean(negs['closer']), np.mean(negs['equal'])
    else:
        negs = {'far': [], 'split': [], 'close' : []}
        with open(click_file, 'r') as f:
            for row in csv.DictReader(f):
                cond = row['condition']
                if 'not' in row['contents']:
                    negs[cond].append(1)
                else:
                    negs[cond].append(0)
        return np.mean(negs['far']), np.mean(negs['split']), np.mean(negs['close'])

def specificity(lang='english'):
    '''
    Returns the average specificity for messages exchanged on each of the
    three conditions (far, split, close) for a given language (English or
    Chinese). Uses WordNet.
    '''
    pass

def boxplot(data, plot_file, xticks=[], xlabel='', ylabel='', title=''):
    fig = plt.figure(1, figsize=(8,6))
    ax = fig.add_subplot(111)
    bp = ax.boxplot(data, patch_artist=True, showmeans=True)
    for box in bp['boxes']:
        box.set(facecolor = '#c7d3d8')
    for median in bp['medians']:
        median.set(color='#a80101')
    for mean in bp['means']:
        mean.set(color='#a80101', marker='.')
    for flier in bp['fliers']:
        flier.set(marker='.', color='#e7298a', alpha=0.5)
    for i in xrange(len(data)):
        d = data[i]
        x, y = bp['medians'][i].get_xydata()[1]
        ax.annotate('median=%.3f,\nmean=%.3f' % (np.median(d), np.mean(d)),
                    (x,y), (x+0.05,0))
    ax.set_xticklabels(xticks)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.savefig(plot_file, bbox_inches='tight', dpi=300)
    plt.gcf().clear()

def show_values(ax, rects):
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height+0.0005,
                '%.3f' % height, ha='center', va='bottom')

# expect list of lists for data
def bargraph(data, plot_file, xticks=[], series_names=[], xlabel='', ylabel='', title=''):
    pos = np.arange(len(data[0]))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8,6))

    rects1 = ax.bar(pos, data[0], width, alpha=0.5, color='#EE3224')
    rects2 = ax.bar(pos+width, data[1], width, alpha=0.5, color='#F78F1E')
    show_values(ax, rects1)
    show_values(ax, rects2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks([p + 0.5 * width for p in pos])
    ax.set_xticklabels(xticks)

    plt.legend(series_names, loc='upper left')
    fig.savefig(plot_file, dpi=300)

def compare(attributeid='', verbose=False, plot=True):
    if attributeid == 'msg_length':
        zh_msg_lengths = msg_lengths(lang='chinese')
        en_msg_lengths = msg_lengths(lang='english')
        if verbose:
            print 'avg msg lengths'
            print ' * ZH: ', np.mean(zh_msg_lengths)
            print ' * EN: ', np.mean(en_msg_lengths)
        if plot:
            boxplot([zh_msg_lengths, en_msg_lengths], 'plots/msg_lengths.png',
            xticks=['Chinese', 'English'], xlabel='Language',
            ylabel='Length of message',
            title='Message lengths for Chinese and English')
    elif attributeid == 'dlg_length':
        zh_dlg_lengths = dlg_lengths(lang='chinese')
        en_dlg_lengths = dlg_lengths(lang='english')
        if verbose:
            print 'avg dialogue lengths'
            print ' * ZH: ', np.mean(zh_dlg_lengths)
            print ' * EN: ', np.mean(en_dlg_lengths)
        if plot:
            boxplot([zh_dlg_lengths, en_dlg_lengths], 'plots/dlg_lengths.png',
            xticks=['Chinese', 'English'], xlabel='Language',
            ylabel='Messages sent per round',
            title='Dialogue lengths for Chinese and English')
    elif attributeid == 'sup_usage':
        zh_sups = superlative_usage(lang='chinese')
        en_sups = superlative_usage(lang='english')
        if verbose:
            print 'percentage of messages using superlatives'
            print ' * ZH: ', zh_sups
            print ' * EN: ', en_sups
        if plot:
            bargraph([list(zh_sups), list(en_sups)], 'plots/sup_usage.png',
                xticks=['further/far', 'closer/split', 'equal/close'],
                series_names=['Chinese', 'English'], xlabel='Condition',
                ylabel='Proportion of messages containing superlative',
                title='Superlative usage for Chinese and English')
    elif attributeid == 'comp_usage':
        zh_comps = comparative_usage(lang='chinese')
        en_comps = comparative_usage(lang='english')
        if verbose:
            print 'percentage of messages using comparatives'
            print ' * ZH: ', zh_comps
            print ' * EN: ', en_comps
        if plot:
            bargraph([list(zh_comps), list(en_comps)], 'plots/comp_usage.png',
                xticks=['further/far', 'closer/split', 'equal/close'],
                series_names=['Chinese', 'English'], xlabel='Condition',
                ylabel='Proportion of messages containing comparative',
                title='Comparative usage for Chinese and English')
    elif attributeid == 'neg_usage':
        zh_negs = negation_usage(lang='chinese')
        en_negs = negation_usage(lang='english')
        if verbose:
            print 'percentage of messages using negation'
            print ' * ZH: ', zh_negs
            print ' * EN: ', en_negs
        if plot:
            bargraph([list(zh_negs), list(en_negs)], 'plots/neg_usage.png',
                xticks=['further/far', 'closer/split', 'equal/close'],
                series_names=['Chinese', 'English'], xlabel='Condition',
                ylabel='Proportion of messages containing negation',
                title='Negation usage for Chinese and English')

if __name__ == '__main__':
    compare('msg_length')
    compare('dlg_length')
    compare('sup_usage')
    compare('comp_usage')
    compare('neg_usage')
