#coding:utf-8

import numpy as np
import enchant
import csv

ZH_MSG_FILE = '../hawkins_data/colorReferenceMessageChinese.csv'
ZH_CLICK_FILE = '../hawkins_data/colorReferenceClicksChinese.csv'
EN_MSG_FILE = '../behavioralAnalysis/humanOutput/filteredCorpus.csv'

PUNCTUATION = ['~',',','.','?','!','。','，']

def avg_msg_length(lang='english'):
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
    return np.mean(msg_lengths)

def avg_dlg_length(lang='english'):
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
    return np.mean(dlg_lengths)

def superlative_usage(lang='english'):
    '''
    Returns how often superlatives are used on each of the three conditions
    (split, far, close) for a given language (English or Chinese).
    '''
    pass

def comparative_usage(lang='english'):
    '''
    Returns how often comparatives are used on each of the three conditions
    (split, far, close) for a given language (English or Chinese).
    '''
    pass

def negation_usage(lang='english'):
    '''
    Returns how often negations are used on each of the three conditions
    (split, far, close) for a given language (English or Chinese).
    '''
    pass

def specificity(lang='english'):
    '''
    Returns the average specificity for messages exchanged on each of the
    three conditions (split, far, close) for a given language (English or
    Chinese). Uses WordNet.
    '''
    pass

if __name__ == '__main__':
    print 'avg msg lengths'
    print avg_msg_length(lang='chinese')
    print avg_msg_length(lang='english')

    print 'avg dialogue lengths'
    print avg_dlg_length(lang='chinese')
    print avg_dlg_length(lang='english')
