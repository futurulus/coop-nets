#coding:utf8
import re
import csv
import numpy as np

from googletrans import Translator
from jieba import tokenize as jieba_tokenize

from nltk.tag import pos_tag
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

############################################################################
# Helper functions for lists, files, printing, etc.
############################################################################

PUNCTUATION = ['~',',','.','?','!','。','，','？','！','”','(',')','…','=']

def flatten(l) :
    return [item for sublist in l for item in sublist]

def dicts_from_file(file_path):
    dicts = []
    with open(file_path, 'r') as f:
        for row in csv.DictReader(f):
            dicts.append(row)
    return dicts

def verbose_msg(heading, zh_data=[], en_data=[]):
    print heading
    print ' * ZH: ', zh_data
    print ' * EN: ', en_data

############################################################################
# Helper functions for message lengths and dialogue lengths
############################################################################

def msg_lengths(msg_rows, L='english'):
    '''
    Returns a list of the length of each message sent for a given language
    (English or Chinese). First, punctuation is removed from the message.
    For English, the message is then tokenized with nltk, and for Chinese,
    the message is tokenized with jieba - but only if it doesn't contain
    any Roman letters.
    '''
    msg_lengths = []
    for row in msg_rows:
        # get msg and remove punctuation (but not apostrophes)
        msg = row['contents']
        for c in PUNCTUATION:
            msg = msg.replace(c, '')
        # disregard any message that is empty or has roman letters
        if L == 'chinese' and msg and not re.search('[a-zA-Z]', msg):
            tokens = jieba_tokenize(unicode(msg.decode('utf8')))
            num_tokens = sum([1 for t in tokens])
            msg_lengths.append(num_tokens)
        elif L == 'english' and msg:
            tokens = word_tokenize(msg)
            msg_lengths.append(len(tokens))
    return msg_lengths

def dlg_lengths(msg_rows):
    '''
    Returns a list of the number of messages exchanged for each round
    for a given language (English or Chinese).
    '''
    counts = {}
    for row in msg_rows:
        roundid = str(row['gameid']) + str(row['roundNum'])
        try:
            counts[roundid] += 1
        except KeyError:
            counts[roundid] = 1
    return counts.values()

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
    if L == 'english':
        tokens = word_tokenize(msg)
        if attribute == 'negation':
            return 'not' in tokens or 'n\'t' in tokens
        else:
            pos_list = pos_tag(tokens)
            if attribute == 'superlative':
                return any([is_superlative(x) for x in pos_list])
            elif attribute == 'comparative':
                return any([is_comparative(x) for x in pos_list])
    elif L == 'chinese':
        if attribute == 'negation':
            return '不' in msg or '没' in msg
        elif attribute == 'superlative':
            return '最' in msg
        elif attribute == 'comparative':
            zh_comps = ['更', '多', '少', '比', '那么']
            return any([x in msg for x in zh_comps])

############################################################################
# Stuff for specificity
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

def specificity(msg, L='english'):
    '''
    Returns the maximal specificity for messages exchanged on each of the
    three conditions (far, split, close). Uses English WordNet and
    Google Translate.
    '''
    en_msg = msg if lang == 'english' else translate(msg)
    depths = [get_informativity(x) for x in en_msg.split()]
    depths = filter(lambda x : x, depths)
    return max(depths) if depths else None
