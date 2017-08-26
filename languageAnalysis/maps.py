def CONDNAME(cond):
    if cond == 'equal' or cond == 'far':
        return 'equal/far'
    elif cond == 'further' or cond == 'split':
        return 'further/split'
    else:
        return 'closer/close'

def LANGNAME(L):
    return 'English' if L == 'en' else 'Chinese'

def ROUNDID(c):
    return '{}.{}'.format(c['gameid'], c['roundNum'])

def YLABEL(attribute):
    if attribute == 'tokens':
        return 'Mean number of tokens per message'
    elif attribute == 'dialogue':
        return 'Mean number of messages exchanged per round'
    elif attribute == 'specificity':
        return 'Mean value of specificity indicator'
    elif attribute == 'success':
        return 'Success rate'
    else:
        return 'Proportion of messages containing %s' % attribute

def PLOTTITLE(attribute):
    if attribute == 'tokens':
        return 'Length of messages for Chinese and English'
    elif attribute == 'dialogue':
        return 'Length of dialogue for Chinese and English'
    elif attribute == 'specificity':
        return 'WordNet specificity for Chinese and English'
    elif attribute == 'success':
        return 'Success rates for Chinese and English'
    else:
        return 'Usage of %s for Chinese and English' % attribute

def PLOTTYPE(attribute):
    return 'bar'
    # if attribute == 'dialogue':
    #     return 'hist'
    # else:
    #     return 'bar'
