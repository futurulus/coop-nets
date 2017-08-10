#coding:utf8
import plots
import utils
import numpy as np
import pandas as pd

ZH_MSG_FILE = 'colorReferenceMessageChinese.csv'
ZH_CLICK_FILE = 'colorReferenceClicksChinese.csv'
EN_FILE = '../behavioralAnalysis/humanOutput/filteredCorpus.csv'
ZH_CONDITIONS = ['equal', 'further', 'closer']
EN_CONDITIONS = ['far', 'split', 'close']

def lengths(attribute='message', L='english'):
    '''
    Returns message lengths or dialogue lengths for the specified language.
    '''
    msg_file = ZH_MSG_FILE if L == 'chinese' else EN_FILE
    msg_rows = utils.dicts_from_file(msg_file)
    if attribute == 'message':
        return utils.msg_lengths(msg_rows, L)
    elif attribute == 'dialogue':
        return utils.dlg_lengths(msg_rows)
    else:
        raise NameError('ATTRIBUTE: try \'message\' or \'dialogue\'.')

def usage(attribute='superlative', L='english'):
    '''
    Returns the proportion of messages that use the specified type of word
    (superlative, comparative, negation) for each of the three conditions
    (far, split, close) in a given language (English or Chinese).
    '''
    click_file = ZH_CLICK_FILE if L == 'chinese' else EN_FILE
    cond_names = ZH_CONDITIONS if L == 'chinese' else EN_CONDITIONS
    counts = {cond : [] for cond in cond_names}
    msg_rows = utils.dicts_from_file(ZH_MSG_FILE)
    click_rows = utils.dicts_from_file(click_file)

    for row in click_rows:
        cond, gameid, roundNum = row['condition'], row['gameid'], row['roundNum']
        if L == 'english':
            msg = row['contents']
            if attribute == 'specificity':
                s = utils.specificity(msg, L)
                if s:
                    counts[cond].append(s)
            else:
                counts[cond].append(int(utils.check_attribute(msg, attribute, L)))
        else:
            round_msgs = [x['contents'] for x in msg_rows
                                        if x['gameid'] == gameid
                                        and x['roundNum'] == roundNum]
            if attribute == 'specificity':
                data = [utils.specificity(msg, L) for msg in round_msgs]
                data = filter(lambda x : x, data)
            else:
                data = [int(utils.check_attribute(msg, attribute, L))
                        for msg in round_msgs]
            counts[cond] += data
    return [np.mean(counts[cond]) for cond in cond_names] # ensures order
    # return [counts[cond] for cond in cond_names]

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
            plots.stripplot([zh_lengths, en_lengths],
                    'plots/%s_lengths_SEABORN.png' % attribute,
                    xticks=['Chinese', 'English'], xlabel='Language',
                    ylabel='Number of %s' % ylabel,
                    title='Length of %s for Chinese and English' % attribute)
    else:
        zh_usage = usage(attribute, 'chinese')
        en_usage = usage(attribute, 'english')
        if verbose:
            if attribute == 'specificity':
                heading = 'Average maximal specificity in all messages:'
            else:
                heading = 'Proportion of messages using %s:' % attribute
            utils.verbose_msg(heading, zh_usage, en_usage)
        if plot:
            if attribute == 'specificity':
                plot_file = 'plots/specificity.png'
                ylabel = 'Average maximal WordNet specificity'
                title = 'WordNet specificity for Chinese and English'
            else:
                plot_file = 'plots/%s_usage.png' % attribute
                ylabel = 'Proportion of messages containing %s' % attribute
                title = 'Usage of %s for Chinese and English' % attribute
            # plots.barplot(pd.DataFrame({'language':['chinese','english'],
            #                             'condition':['equal/far', 'further/split', 'closer/close'],
            #                             'usages':[zh_usage, en_usage]}))
            plots.bargraph([list(zh_usage), list(en_usage)], plot_file,
                    xticks=['equal/far', 'further/split', 'closer/close'],
                    series_names=['Chinese', 'English'], xlabel='Condition',
                    ylabel=ylabel, title=title)

if __name__ == '__main__':
    # compare('message')
    # compare('dialogue')
    # compare('superlative')
    # compare('comparative')
    # compare('negation')
    # compare('specificity')
