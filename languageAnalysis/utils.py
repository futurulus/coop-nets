#coding:utf8
import re
import csv
import plots
from maps import *
import pandas as pd
from specificity import *
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from jieba import tokenize as jieba_tokenize

############################################################################
# Helper functions for files, plotting, etc.
############################################################################

def dicts_from_file(infile):
    dicts = []
    with open(infile, 'r') as f:
        for row in csv.DictReader(f):
            dicts.append(row)
    return dicts

def list_to_csv(data, outfile, col_names):
    with open(outfile, 'wb') as f:
        w = csv.writer(f)
        w.writerow(col_names)
        for row in data:
            w.writerow(row)

def generate_csv(attribute, L):
    print 'Generating csv for {} ({})...'.format(attribute, L)
    data = get_data(attribute, L)
    s = 'roundid' if attribute == 'dialogue' \
                  or attribute == 'success' else 'Message'
    col_names = [attribute, 'Condition', 'Language', s]
    list_to_csv(data, 'data/{}_{}.csv'.format(attribute, L), col_names)

def plot_csvs(zh_file, en_file, plot_type, plot_file, ylabel, title):
    zh_df = pd.read_csv(zh_file)
    en_df = pd.read_csv(en_file)
    df = zh_df.append(en_df)
    plot_fun = plots.histogram if plot_type == 'hist' else plots.barplot
    df.drop(df.columns[-1], axis=1, inplace=True)
    plot_fun(df, plot_file, ylabel, title)

def plot_for_attribute(a):
    f_zh = 'data/{}_zh.csv'.format(a)
    f_en = 'data/{}_en.csv'.format(a)
    print 'Plotting for {}...'.format(a)
    plot_csvs(f_zh, f_en, PLOTTYPE(a), 'plots/%s.png' % a,
              ylabel=YLABEL(a), title=PLOTTITLE(a))

############################################################################
# Helper functions for checking superlatives, comparatives, and negations
############################################################################

def is_superlative(lemma):
    return lemma[1] == 'JJS' or lemma[1] == 'RBS' \
           or (lemma[1] == 'NN' and lemma[0][-3:] == 'est' \
                                and lemma[0] != 'forest')

def is_comparative(lemma):
    er_nouns = ['other', 'speaker', 'listener', 'partner', 'summer', 'closer',
                'flower', 'hunter', 'water', 'together', 'copper', 'lavender']
    return lemma[1] == 'JJR' or lemma[1] == 'RBR' \
           or (lemma[1] == 'NN' and lemma[0][-2:] == 'er' \
                                and lemma[0] not in er_nouns)

def check_attribute(msg, attribute, L):
    '''
    Returns true if the given attribute (superlative, comparative, negation)
    is present in the message for a given language (English or Chinese).
    '''
    if L == 'en':
        tokens = word_tokenize(msg)
        if attribute == 'negation':
            return 'not' in tokens or 'n\'t' in tokens
        else:
            pos_list = pos_tag(tokens)
            if attribute == 'superlative':
                return any([is_superlative(x) for x in pos_list])
            elif attribute == 'comparative':
                return any([is_comparative(x) for x in pos_list])
    elif L == 'zh':
        if attribute == 'negation':
            return '不' in msg or '没' in msg
        elif attribute == 'superlative':
            return '最' in msg
        elif attribute == 'comparative':
            return '更' in msg or '比' in msg
            # return ('多' in msg and '最' not in msg) \
            #         or ('少' in msg and '最' not in msg)

############################################################################
# Helper functions for getting data for message- and round-based attributes
############################################################################

PUNCTUATION = ['~', ',', '.', '?', '!', '。', '，', '、', ':',
               '？', '！', '”', '(', ')', '…', '=', '-', '_', '～']

def update_message_data(data, msg, cond, attribute, L):
    x = None
    if attribute == 'tokens':
        for p in PUNCTUATION:
            msg = msg.replace(p, '')
        # disregard any message that is empty or has roman letters
        if L == 'zh' and msg and not re.search('[a-zA-Z]', msg):
            tokens = list(jieba_tokenize(unicode(msg.decode('utf8'))))
            x = len(tokens)
        elif L == 'en' and msg:
            tokens = word_tokenize(msg)
            x = len(tokens)
    elif attribute == 'specificity':
        x = specificity(msg, L)
    else:
        x = check_attribute(msg, attribute, L)
    if x is not None:
        data.append([x, CONDNAME(cond), LANGNAME(L), msg])

def message_data(attribute, L, msg_dicts, click_dicts):
    '''
    Returns the proportion of messages that use the specified type of word
    (superlative, comparative, negation) for each of the three conditions
    (far, split, close) in a given language (English or Chinese).
    '''
    data = []
    for c in click_dicts:
        cond, gameid, roundNum = c['condition'], c['gameid'], c['roundNum']
        if L == 'en':
            update_message_data(data, c['contents'], cond, attribute, L)
        elif L == 'zh':
            for m in msg_dicts:
                if m['gameid'] == gameid and m['roundNum'] == roundNum:
                    update_message_data(data, m['contents'], cond, attribute, L)
    return data

def update_round_data(counts, conds, cond, roundid):
    conds[roundid] = CONDNAME(cond)
    counts[roundid] = counts[roundid] + 1 if roundid in counts.keys() else 1

def round_data(attribute, L, msg_dicts, click_dicts):
    counts, conds = {}, {}
    for c in click_dicts:
        cond, roundid = c['condition'], ROUNDID(c)
        if attribute == 'success':
            counts[roundid] = c['outcome']
            conds[roundid] = CONDNAME(cond)
        elif attribute == 'dialogue':
            if L == 'en':
                update_round_data(counts, conds, cond, roundid)
            elif L == 'zh':
                for m in msg_dicts:
                    if m['gameid'] == c['gameid'] \
                    and m['roundNum'] == c['roundNum']:
                        update_round_data(counts, conds, cond, roundid)
    return [[counts[roundid], conds[roundid], LANGNAME(L), roundid]
            for roundid in sorted(counts)]

############################################################################
# The final get_data function
############################################################################

ZH_MSG_FILE = 'data_input_cleaned/colorReferenceMessageChinese_filtered.csv'
ZH_CLICK_FILE = 'data_input_raw/colorReferenceClicksChinese.csv'
EN_FILE = '../behavioralAnalysis/humanOutput/filteredCorpus.csv'

def get_data(attribute, L):
    click_file = ZH_CLICK_FILE if L == 'zh' else EN_FILE
    msg_dicts = dicts_from_file(ZH_MSG_FILE)
    click_dicts = dicts_from_file(click_file)

    if attribute == 'dialogue' or attribute == 'success':
        return round_data(attribute, L, msg_dicts, click_dicts)
    else:
        return message_data(attribute, L, msg_dicts, click_dicts)
