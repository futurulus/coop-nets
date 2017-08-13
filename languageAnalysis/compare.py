#coding:utf8
import plots
import utils
import numpy as np
import pandas as pd

ZH_MSG_FILE = 'colorReferenceMessageChinese.csv'
ZH_CLICK_FILE = 'colorReferenceClicksChinese.csv'
EN_FILE = '../behavioralAnalysis/humanOutput/filteredCorpus.csv'

def lengths(attribute='message', L='en'):
    '''
    Returns message lengths or dialogue lengths for the specified language.
    '''
    msg_file = ZH_MSG_FILE if L == 'zh' else EN_FILE
    msg_dicts = utils.dicts_from_file(msg_file)
    if attribute == 'message':
        return utils.msg_lengths(msg_dicts, L)
    elif attribute == 'dialogue':
        return utils.dlg_lengths(msg_dicts, L)
    else:
        raise NameError('ATTRIBUTE: try \'message\' or \'dialogue\'.')

def usage(attribute='superlative', L='en'):
    '''
    Returns the proportion of messages that use the specified type of word
    (superlative, comparative, negation) for each of the three conditions
    (far, split, close) in a given language (English or Chinese).
    '''
    click_file = ZH_CLICK_FILE if L == 'zh' else EN_FILE
    msg_dicts = utils.dicts_from_file(ZH_MSG_FILE)
    click_dicts = utils.dicts_from_file(click_file)

    data = []
    for c in click_dicts:
        cond, gameid, roundNum = c['condition'], c['gameid'], c['roundNum']
        if L == 'en':
            utils.update(data, c['contents'], cond, attribute, L)
        else:
            for m in msg_dicts:
                if m['gameid'] == gameid and m['roundNum'] == roundNum:
                    utils.update(data, m['contents'], cond, attribute, L)
    return data

def compare(attribute, verbose=True, plot=True):
    if attribute == 'message' or attribute == 'dialogue':
        zh_lengths = lengths(attribute, 'zh')
        en_lengths = lengths(attribute, 'en')
        data = zh_lengths + en_lengths
        thirdcol = 'Messsage' if attribute == 'message' else 'roundid'
        df = pd.DataFrame(data, columns=['Length', 'Language', thirdcol])
        df.to_csv('data/%s_lengths.csv' % attribute)

        if verbose:
            print 'Done finding %s lengths.' % attribute
        if plot:
            if attribute == 'message':
                ylabel = 'Number of words per message'
            else:
                ylabel = 'Number of messages sent per round'
            df.drop(thirdcol, axis=1, inplace=True)
            plots.boxplot(df, 'plots/%s_lengths_SEABORN.png' % attribute,
                          ylabel=ylabel, strip=True,
                          title='Length of %s for Chinese and English' % attribute)

    else:
        zh_usage = usage(attribute, 'zh')
        en_usage = usage(attribute, 'en')
        data = zh_usage + en_usage
        df = pd.DataFrame(data, columns=['Usage', 'Condition', 'Language', 'Message'])

        if verbose:
            print 'Done comparing %s.' % attribute
        if plot:
            if attribute == 'specificity':
                df.to_csv('data/specificity.csv')
                plot_file = 'plots/specificity_SEABORN.png'
                ylabel = 'Average maximal WordNet specificity'
                title = 'WordNet specificity for Chinese and English'
            else:
                df.to_csv('data/%s_usage.csv' % attribute)
                plot_file = 'plots/%s_usage_SEABORN.png' % attribute
                ylabel = 'Proportion of messages containing %s' % attribute
                title = 'Usage of %s for Chinese and English' % attribute
            df.drop('Message', axis=1, inplace=True)
            plots.barplot(df, plot_file, ylabel, title)

if __name__ == '__main__':
    # compare('message')
    # compare('dialogue')
    compare('superlative')
    compare('comparative')
    compare('negation')
    compare('specificity')
