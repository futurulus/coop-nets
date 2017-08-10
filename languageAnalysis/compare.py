#coding:utf-8
import plots
import utils
import numpy as np
import enchant
import csv

ZH_MSG_FILE = 'colorReferenceMessageChinese.csv'
ZH_CLICK_FILE = 'colorReferenceClicksChinese.csv'
ZH_CONDITIONS = ['equal', 'further', 'closer']
EN_FILE = '../behavioralAnalysis/humanOutput/filteredCorpus.csv'
EN_CONDITIONS = ['far', 'split', 'close']
PUNCTUATION = ['~',',','.','?','!','。','，','？','！','”','(',')','…','=']

def msg_lengths(lang='english'):
    '''
    Returns the lengths of the messages sent for a given language
    (English or Chinese). For English, it counts the number of words
    (after removing punctuation and splitting on whitespaces), and for Chinese,
    it counts the number of chars (after removing punctuation, whitespaces,
    and any messages containing English).
    '''
    msg_file = ZH_MSG_FILE if lang == 'chinese' else EN_FILE
    d = enchant.Dict('en_US')
    msg_lengths = []
    with open(msg_file, 'r') as f:
        for row in csv.DictReader(f):
            msg = row['contents']
            for c in PUNCTUATION:
                msg = msg.replace(c, '')
            if lang == 'chinese':
                msg = '' if any([d.check(s) for s in msg.split()]) \
                        else msg.replace(' ', '')
                msg_lengths.append(len(msg.decode('utf8')))
            else:
                msg_lengths.append(len(msg.split()))
    return msg_lengths

def dlg_lengths(lang='english'):
    '''
    Returns the numbers of messages exchanged for each round
    for a given language (English or Chinese).
    '''
    msg_file = ZH_MSG_FILE if lang == 'chinese' else EN_FILE
    counts = {}
    with open(msg_file, 'r') as f:
        for row in csv.DictReader(f):
            roundid = str(row['gameid']) + str(row['roundNum'])
            try:
                counts[roundid] += 1
            except KeyError:
                counts[roundid] = 1
    return counts.values()

def lengths(attribute='message', lang='english'):
    '''
    Returns message lengths or dialogue lengths for the specified language.
    '''
    if attribute == 'message':
        return msg_lengths(lang)
    elif attribute == 'dialogue':
        return dlg_lengths(lang)
    else:
        raise NameError('ATTRIBUTE: try \'message\' or \'dialogue\'.')

def usage(attribute='superlative', lang='english'):
    '''
    Returns the proportion of messages that use the specified type of word
    (superlative, comparative, negation) for each of the three conditions
    (far, split, close) in a given language (English or Chinese).
    '''
    click_file = ZH_CLICK_FILE if lang == 'chinese' else EN_FILE
    cond_names = ZH_CONDITIONS if lang == 'chinese' else EN_CONDITIONS
    counts = {cond : [] for cond in cond_names}
    msg_rows = utils.dicts_from_file(ZH_MSG_FILE)
    click_rows = utils.dicts_from_file(click_file)

    for row in click_rows:
        cond, gameid, roundNum = row['condition'], row['gameid'], row['roundNum']
        if lang == 'english':
            if attribute == 'specificity':
                s = utils.specificity(row['contents'])
                if s:
                    counts[cond].append(s)
            else:
                counts[cond].append(int(utils.attribute_in_msg(row['contents'],
                                    attribute, lang)))
        else:
            round_msgs = [x['contents'] for x in msg_rows
                                        if x['gameid'] == gameid
                                        and x['roundNum'] == roundNum]
            if attribute == 'specificity':
                data = [utils.specificity(msg) for msg in round_msgs
                                               if utils.specificity(msg)]
            else:
                data = [int(utils.attribute_in_msg(msg, attribute, lang))
                        for msg in round_msgs]
            counts[cond] += data
    return [np.mean(counts[cond]) for cond in cond_names] # ensures order

def compare(attribute, verbose=True, plot=True):
    if attribute == 'message' or attribute == 'dialogue':
        zh_lengths = lengths(attribute, 'chinese')
        en_lengths = lengths(attribute, 'english')
        ylabel = 'words per message' if attribute == 'message' \
                                     else 'messages sent per round'
        if verbose:
            utils.verbose_msg('Average %s lengths:' % attribute,
                        np.mean(zh_lengths), np.mean(en_lengths))
        if plot:
            plots.boxplot([zh_lengths, en_lengths],
                    'plots/%s_lengths.png' % attribute,
                    xticks=['Chinese', 'English'], xlabel='Language',
                    ylabel='Number of %s' % ylabel,
                    title='Length of %s for Chinese and English' % attribute)
    else:
        # zh_usage = usage(attribute, 'chinese')
        zh_usage = None
        en_usage = usage(attribute, 'english')
        if verbose:
            heading = None
            if attribute == 'specificity':
                heading = 'Average maximal specificity in all messages:'
            else:
                heading = 'Proportion of messages using %s:' % attribute
            utils.verbose_msg(heading, zh_usage, en_usage)
        # if plot:
        #     plot_file, ylabel, title = None, None, None
        #     if attribute == 'specificity':
        #         plot_file = 'plots/specificity.png'
        #         ylabel = 'Average maximal WordNet specificity'
        #         title = 'WordNet specificity for Chinese and English'
        #     else:
        #         plot_file = 'plots/%s_usage.png' % attribute
        #         ylabel = 'Proportion of messages containing %s' % attribute
        #         title = 'Usage of %s for Chinese and English' % attribute
        #     plots.bargraph([list(zh_usage), list(en_usage)], plot_file,
        #             xticks=['equal/far', 'further/split', 'closer/close'],
        #             series_names=['Chinese', 'English'], xlabel='Condition',
        #             ylabel=ylabel, title=title)

if __name__ == '__main__':
    # compare('message')
    # compare('dialogue')
    # compare('superlative')
    # compare('comparative')
    # compare('negation')
    compare('specificity')

    # zh_usage = [6.7184466019417473, 6.6880341880341883, 6.7615384615384615]
    # [7.035610710190868, 7.0520575740282796, 7.0389788293897881]
    #
    # plot_file = 'plots/specificity.png'
    # ylabel = 'Average specificity'
    # title = 'WordNet specificity for Chinese and English'
    # plots.bargraph([zh_usage, en_usage], plot_file,
    #         xticks=['equal/far', 'further/split', 'closer/close'],
    #         series_names=['Chinese', 'English'], xlabel='Condition',
    #         ylabel=ylabel, title=title)

    # print specificity('color')
    # print specificity('red')
    # print specificity('crimson')
    # print specificity('the')
    # print specificity('grass')
    # print specificity('this color is a bright turquoise')
    # print specificity('blue')
    # print specificity('绿')
    # print specificity('草')
    # print specificity('接近草的颜色，可是没那么绿')
