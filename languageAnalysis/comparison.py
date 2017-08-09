#coding:utf-8

import plots
import utils
import numpy as np
import enchant
import csv

ZH_MSG_FILE = 'colorReferenceMessageChinese.csv'
ZH_CLICK_FILE = 'colorReferenceClicksChinese.csv'
ZH_CONDITIONS = ['further', 'closer', 'equal']
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
    if lang == 'chinese':
        msg_file = ZH_MSG_FILE
        d = enchant.Dict('en_US')
    else:
        msg_file = EN_FILE
    msg_lengths = []
    with open(msg_file, 'r') as f:
        for row in csv.DictReader(f):
            msg = row['contents']
            # remove punctuation marks (primarily for chinese)
            for c in PUNCTUATION:
                msg = msg.replace(c, '')
            if lang == 'chinese':
                # get rid of all messages containing english and remove spaces
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
            counts[cond].append(int(utils.attribute_in_msg(row['contents'],
                                attribute, lang)))
        else:
            round_msgs = [x['contents'] for x in msg_rows
                                        if x['gameid'] == gameid
                                        and x['roundNum'] == roundNum]
            counts[cond] += [int(utils.attribute_in_msg(msg, attribute, lang))
                            for msg in round_msgs]
    return [np.mean(counts[cond]) for cond in cond_names] # ensures order

def specificity(lang='english'):
    '''
    Returns the average specificity for messages exchanged on each of the
    three conditions (far, split, close) for a given language (English or
    Chinese). Uses WordNet.
    '''
    pass

def compare(metric, attribute, verbose=True, plot=True):
    if metric == 'length':
        zh_lengths = lengths(attribute, 'chinese')
        en_lengths = lengths(attribute, 'english')
        ylabel = 'words per message' if attribute == 'message' \
                                     else 'messages sent per round'
        if verbose:
            print 'Average %s lengths:' % attribute
            print ' * ZH: ', np.mean(zh_lengths)
            print ' * EN: ', np.mean(en_lengths)
        if plot:
            plots.boxplot([zh_lengths, en_lengths],
                    'plots/%s_lengths.png' % attribute,
                    xticks=['Chinese', 'English'], xlabel='Language',
                    ylabel='Number of %s' % ylabel,
                    title='Length of %s for Chinese and English' % attribute)
    elif metric == 'usage':
        zh_usage = usage(attribute, 'chinese')
        en_usage = usage(attribute, 'english')
        if verbose:
            print 'Proportion of messages using %s:' % attribute
            print ' * ZH: ', zh_usage
            print ' * EN: ', en_usage
        if plot:
            plots.bargraph([list(zh_usage), list(en_usage)],
                    'plots/%s_usage.png' % attribute,
                    xticks=['further/far', 'closer/split', 'equal/close'],
                    series_names=['Chinese', 'English'], xlabel='Condition',
                    ylabel='Proportion of messages containing %s' % attribute,
                    title='Usage of %s for Chinese and English' % attribute)
    else:
        raise NameError('Invalid metric: try \'length\' or \'usage\'.')

if __name__ == '__main__':
    compare('length', 'message')
    compare('length', 'dialogue')
    compare('usage', 'superlative')
    compare('usage', 'comparative')
    compare('usage', 'negation')
