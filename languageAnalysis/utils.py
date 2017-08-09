#coding:utf8
import csv

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
