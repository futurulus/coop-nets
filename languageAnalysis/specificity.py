from googletrans import Translator
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

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
    return max(res) if res else None

def translate(msg, dest='en'):
    '''
    Translates a string into the target language using Google Translate.
    Defaults to English as target.
    '''
    return Translator().translate(msg, dest=dest).text

def specificity(msg, L):
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
