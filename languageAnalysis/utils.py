#coding:utf8
import csv
import numpy as np
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def dicts_from_file(file_path):
    '''
    Returns a list of dictionaries read from a csv file.
    '''
    dicts = []
    with open(file_path, 'r') as f:
        for row in csv.DictReader(f):
            dicts.append(row)
    return dicts

def attribute_in_msg(msg, attribute, lang):
    '''
    Returns true if the given attribute (superlative, comparative, negation)
    is present in the message for a given language (English or Chinese).
    '''
    if attribute == 'superlative':
        if lang == 'english':
            return 'est' in msg \
                    or ('most' in msg and 'almost' not in msg)
        elif lang == 'chinese':
            return '最' in msg
    elif attribute == 'comparative':
        if lang == 'english':
            # import nltk
            # text = nltk.word_tokenize(msg)
            # pos_list = nltk.pos_tag(text)
            # try:
            #     adj = next(x for x in pos_list if x[1] == 'JJ')
            #     return 'er' in adj[0]
            # except StopIteration:
            #     return False
            er_words = ['other', 'hunter', 'water', 'different', 'wonderful',
                        'partner', 'forever', 'there', 'were', 'are',
                        'periwinkle', 'every', 'lavender', 'copper', 'berry',
                        'very', 'where', 'person', 'here', 'speaker', 'silver',
                        'listener', 'over']
            return ('er' in msg and all([x not in msg.lower() for x in er_words])) \
                    or 'more' in msg or 'less ' in msg
        elif lang == 'chinese':
            zh_comps = ['更', '多', '少', '比', '那么']
            return any([x in msg for x in zh_comps])
    elif attribute == 'negation':
        if lang == 'english':
            return 'not' in msg and 'another' not in msg
        elif lang == 'chinese':
            return '不' in msg or '没' in msg
    else:
        raise NameError('ATTRIBUTE: try \'superlative\', \'comparative\', or \'negation\'.')

def flatten(l) :
    return [item for sublist in l for item in sublist]

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

def translate(target, msg):
    '''
    Translates a given string into the specified target language.
    Uses Google Translate.
    '''
    from googletrans import Translator
    translator = Translator()
    return translator.translate(msg, dest=target).text

def specificity(msg):
    '''
    Returns the maximal specificity for messages exchanged on each of the
    three conditions (far, split, close). Uses English WordNet and
    Google Translate.
    '''
    # en_msg = translate('en', msg)
    en_msg = msg
    informativities = [get_informativity(x) for x in en_msg.split()]
    cleaned = filter(lambda x : x, informativities)
    return max(cleaned) if cleaned else None

def verbose_msg(heading, zh_data=[], en_data=[]):
    print heading
    print ' * ZH: ', zh_data
    print ' * EN: ', en_data
