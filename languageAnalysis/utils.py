#coding:utf8
import re
import csv
import plots
import numpy as np
import pandas as pd

from googletrans import Translator
from jieba import tokenize as jieba_tokenize

from nltk.tag import pos_tag
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

############################################################################
# Helper functions for lists, files, printing, etc.
############################################################################

def flatten(l) :
    return [item for sublist in l for item in sublist]

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
    data = col_names = None
    if attribute == 'dialogue':
        data = lengths(attribute, L)
        col_names = ['Length', 'Language', 'roundid']
    else:
        data = usage(attribute, L)
        col_names = [attribute, 'Condition', 'Language', 'Message']
    list_to_csv(data, 'data/{}_{}.csv'.format(attribute, L), col_names)

def plot_csvs(zh_file, en_file, plot_type, plot_file, ylabel, title):
    zh_df = pd.read_csv(zh_file)
    en_df = pd.read_csv(en_file)
    df = zh_df.append(en_df)
    col_to_drop = df.columns[2] if plot_type == 'hist' else 'Message'
    plot_fun = plots.histogram if plot_type == 'hist' else plots.barplot
    df.drop(col_to_drop, axis=1, inplace=True)
    plot_fun(df, plot_file, ylabel, title)

############################################################################
# Helper functions for mapping keys to values
############################################################################

def CONDNAME(cond):
    if cond == 'equal' or cond == 'far':
        return 'equal/far'
    elif cond == 'further' or cond == 'split':
        return 'further/split'
    else:
        return 'closer/close'

def LANGNAME(L):
    return 'English' if L == 'en' else 'Chinese'

def YLABEL(attribute):
    if attribute == 'tokens':
        return 'Number of tokens per message'
    elif attribute == 'dialogue':
        return 'Number of messages exchanged per round'
    else:
        return 'Proportion of messages containing %s' % attribute

def PLOTTITLE(attribute):
    if attribute == 'tokens':
        return 'Length of messages for Chinese and English'
    elif attribute == 'dialogue':
        return 'Length of dialogue for Chinese and English'
    else:
        return 'Usage of %s for Chinese and English' % attribute

def PLOTTYPE(attribute):
    if attribute == 'dialogue':
        return 'hist'
    else:
        return 'bar'

############################################################################
# Helper functions for message lengths and dialogue lengths
############################################################################

# def msg_lengths(msg_dicts, L='en'):
#     '''
#     Returns a list of the length of each message sent for a given language
#     (English or Chinese). First, punctuation is removed from the message.
#     For English, the message is then tokenized with nltk, and for Chinese,
#     the message is tokenized with jieba - but only if it doesn't contain
#     any Roman letters.
#     '''
#     data = []
#     for m in msg_dicts:
#         # get msg and remove punctuation (but not apostrophes)
#         msg = m['contents']
#         for p in PUNCTUATION:
#             msg = msg.replace(p, '')
#         # disregard any message that is empty or has roman letters
#         if L == 'zh' and msg and not re.search('[a-zA-Z]', msg):
#             tokens = list(jieba_tokenize(unicode(msg.decode('utf8'))))
#             data.append([len(tokens), LANGNAME(L), msg])
#         elif L == 'en' and msg:
#             tokens = word_tokenize(msg)
#             data.append([len(tokens), LANGNAME(L), msg])
#     return data

PUNCTUATION = ['~',',','.','?','!','。','，','、',':',
                '？','！','”','(',')','…','=','-','_','～']

def dlg_lengths(msg_dicts, L='en'):
    '''
    Returns a list of the number of messages exchanged for each round
    for a given language (English or Chinese).
    '''
    counts = {}
    for m in msg_dicts:
        roundid = '{}.{}'.format(m['gameid'], m['roundNum'])
        try:
            counts[roundid] += 1
        except KeyError:
            counts[roundid] = 1
    return [[counts[roundid], LANGNAME(L), roundid] for roundid in counts.keys()]

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
            if '更' in msg or '比' in msg:
                return True
            else:
                return ('多' in msg and '最' not in msg) \
                        or ('少' in msg and '最' not in msg)
            # zh_comps = ['更', '多', '少', '比']
            # return any([x in msg for x in zh_comps])

def update(data, msg, cond, attribute='superlative', L='en'):
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
        x = int(check_attribute(msg, attribute, L))
    if x is not None:
        data.append([x, CONDNAME(cond), LANGNAME(L), msg])

############################################################################
# Helper functions for checking superlatives, comparatives, and negations
############################################################################

ZH_MSG_FILE = 'colorReferenceMessageChinese.csv'
ZH_CLICK_FILE = 'colorReferenceClicksChinese.csv'
EN_FILE = '../behavioralAnalysis/humanOutput/filteredCorpus.csv'

def usage(attribute='superlative', L='en'):
    '''
    Returns the proportion of messages that use the specified type of word
    (superlative, comparative, negation) for each of the three conditions
    (far, split, close) in a given language (English or Chinese).
    '''
    click_file = ZH_CLICK_FILE if L == 'zh' else EN_FILE
    msg_dicts = dicts_from_file(ZH_MSG_FILE)
    click_dicts = dicts_from_file(click_file)

    data = []
    for c in click_dicts:
        cond, gameid, roundNum = c['condition'], c['gameid'], c['roundNum']
        if L == 'en':
            update(data, c['contents'], cond, attribute, L)
        else:
            for m in msg_dicts:
                if m['gameid'] == gameid and m['roundNum'] == roundNum:
                    update(data, m['contents'], cond, attribute, L)
    return data

############################################################################
# Helper functions for specificity
############################################################################

def nounify(adj_word):
    """ Transform an adjective to the closest noun: dead -> death """
    adj_synsets = wn.synsets(adj_word, pos='a')
    # Word not found
    if not adj_synsets:
        return []
    # Get all adj lemmas of the word
    adj_lemmas = [l for s in adj_synsets
                  for l in s.lemmas()
                  if (s.name().split('.')[1] == 'a' or
                      s.name().split('.')[1] == 's')]
    # Get related forms
    derivationally_related_forms = [(l, l.derivationally_related_forms())
                                    for l in adj_lemmas]
    # filter only the nouns
    related_noun_lemmas = [l for drf in derivationally_related_forms
                           for l in drf[1]
                           if l.synset().name().split('.')[1] == 'n']
    synsets = [l.synset() for l in related_noun_lemmas]
    return synsets

def get_informativity(text):
    wnl = WordNetLemmatizer()
    try:
        words = [wnl.lemmatize(word) for word in word_tokenize(text)]
    except:
        print(text)
        raise
    res = []
    for word in words:
        nounForms = wn.synsets(word, pos='n')
        nounSynsets = nounForms if nounForms else nounify(word)
        colorSynsets = [n for n in nounSynsets
                        if 'color.n.01' in
                        [s.name() for s in flatten(n.hypernym_paths())]]
        res += [s.min_depth() for s in colorSynsets][:1] if colorSynsets else []
    return np.max(res) if res else None

def translate(msg, dest='en'):
    '''
    Translates a string into the target language using Google Translate.
    Defaults to English as target.
    '''
    return Translator().translate(msg, dest=dest).text

def specificity(msg, L='en'):
    '''
    Returns the maximal specificity for messages exchanged on each of the
    three conditions (far, split, close). Uses English WordNet and
    Google Translate.
    '''
    en_msg = msg if L == 'en' else translate(msg)
    depths = [get_informativity(x) for x in en_msg.split()]
    depths = filter(lambda x : x, depths)
    if depths:
        return 1 if max(depths) > 7 else 0
    else:
        return None
