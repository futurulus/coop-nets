# coding:utf-8

from collections import defaultdict, Counter
from numbers import Number
import numpy as np
import math
from nltk.corpus import wordnet as wn

# for JennsLearner
from sklearn.linear_model import LogisticRegression
from scipy.misc import logsumexp
from nltk.corpus import stopwords
from collections import Counter

from stanza.monitoring import progress
from stanza.research.learner import Learner
from stanza.research import config
from lux import LuxLearner
from listener import LISTENERS
from speaker import SPEAKERS
from vectorizers import BucketsVectorizer, SequenceVectorizer, FourierVectorizer
from tokenizers import TOKENIZERS
from rsa import RSALearner


def new(key):
    '''
    Construct a new learner with the class named by `key`. A list
    of available learners is in the dictionary `LEARNERS`.
    '''
    return LEARNERS[key]()


class Histogram(object):
    '''
    >>> from stanza.research.instance import Instance as I
    >>> data = [I((0.0, 100.0, 49.0), 'red'),
    ...         I((0.0, 100.0, 45.0), 'dark red'),
    ...         I((240.0, 100.0, 49.0), 'blue')]
    >>> h = Histogram(data, names=['red', 'dark red', 'blue'],
    ...               granularity=(4, 10, 10))
    >>> h.get_probs((1.0, 91.0, 48.0))
    [0.5, 0.5, 0.0]
    >>> h.get_probs((240.0, 100.0, 40.0))
    [0.0, 0.0, 1.0]
    '''
    def __init__(self, training_instances, names,
                 granularity=(1, 1, 1), use_progress=False):
        self.names = names
        self.buckets = defaultdict(Counter)
        self.bucket_counts = defaultdict(int)
        self.granularity = granularity
        self.bucket_sizes = (360 // granularity[0],
                             100 // granularity[1],
                             100 // granularity[2])
        self.use_progress = use_progress

        self.add_data(training_instances)

    def add_data(self, training_instances):
        if self.use_progress:
            progress.start_task('Example', len(training_instances))

        for i, inst in enumerate(training_instances):
            if self.use_progress:
                progress.progress(i)

            bucket = self.get_bucket(inst.input)
            self.buckets[bucket][inst.output] += 1
            self.bucket_counts[bucket] += 1

        if self.use_progress:
            progress.end_task()

    def get_bucket(self, color):
        '''
        >>> Histogram([], [], granularity=(3, 5, 10)).get_bucket((0, 1, 2))
        (0, 0, 0)
        >>> Histogram([], [], granularity=(3, 5, 10)).get_bucket((172.0, 30.0, 75.0))
        (120, 20, 70)
        >>> Histogram([], [], granularity=(3, 5, 10)).get_bucket((360.0, 100.0, 100.0))
        (240, 80, 90)
        '''
        return tuple(
            s * min(int(d // s), g - 1)
            for d, s, g in zip(color, self.bucket_sizes, self.granularity)
        )

    def get_probs(self, color):
        bucket = self.get_bucket(color)
        counter = self.buckets[bucket]
        bucket_size = self.bucket_counts[bucket]
        probs = []
        for name in self.names:
            prob = ((counter[name] * 1.0 / bucket_size)
                    if bucket_size != 0
                    else (1.0 / len(self.names)))
            probs.append(prob)
        return probs

    @property
    def num_params(self):
        return sum(len(counter) for _name, counter in self.buckets.items())

    def __getstate__(self):
        # `defaultdict`s aren't pickleable. Turn them into regular dicts for pickling.
        state = dict(self.__dict__)
        for name in ('buckets', 'bucket_counts'):
            state[name] = dict(state[name])
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.buckets = defaultdict(Counter, self.buckets)
        self.bucket_counts = defaultdict(int, self.bucket_counts)


class HistogramLearner(Learner):
    '''
    The histogram model (HM) baseline from section 5.1 of McMahan and Stone
    (2015).
    '''

    WEIGHTS = [0.322, 0.643, 0.035]
    GRANULARITY = [(90, 10, 10), (45, 5, 5), (1, 1, 1)]

    def __init__(self):
        self.hists = []
        self.names = []
        self.name_to_index = defaultdict(lambda: -1)

    def train(self, training_instances, validation_instances='ignored', metrics='ignored'):
        self.names = sorted(set(inst.output for inst in training_instances)) + ['<unk>']
        self.name_to_index = defaultdict(lambda: -1,
                                         {n: i for i, n in enumerate(self.names)})
        self.hists = []
        progress.start_task('Histogram', len(self.GRANULARITY))
        for i, g in enumerate(self.GRANULARITY):
            progress.progress(i)
            self.hists.append(Histogram(training_instances, self.names,
                                        granularity=g, use_progress=True))
        progress.end_task()

        self.num_params = sum(h.num_params for h in self.hists)

    def hist_probs(self, color):
        assert self.hists, \
            'No histograms constructed yet; calling predict/score before train?'

        probs = [np.array(h.get_probs(color)) for h in self.hists]
        return sum(w * p for w, p in zip(self.WEIGHTS, probs))

    def predict_and_score(self, eval_instances):
        predictions = []
        scores = []
        progress.start_task('Example', len(eval_instances))
        for i, inst in enumerate(eval_instances):
            progress.progress(i)
            hist_probs = self.hist_probs(inst.input)
            name = self.names[hist_probs.argmax()]
            prob = hist_probs[self.name_to_index[inst.output]]
            predictions.append(name)
            scores.append(np.log(prob))
        progress.end_task()
        return predictions, scores

    def __getstate__(self):
        state = dict(self.__dict__)
        state['name_to_index'] = dict(state['name_to_index'])
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.name_to_index = defaultdict(lambda: -1, self.name_to_index)


class MostCommonSpeakerLearner(Learner):
    def __init__(self):
        self.seen = Counter()
        self.num_examples = 0

    def train(self, training_instances, validation_instances='ignored', metrics='ignored'):
        progress.start_task('Example', len(training_instances))
        for i, inst in enumerate(training_instances):
            progress.progress(i)
            self.seen.update([inst.output])
        progress.end_task()
        self.num_examples += len(training_instances)

    @property
    def num_params(self):
        return len(self.seen)

    def predict_and_score(self, eval_instances):
        most_common = self.seen.most_common(1)[0][0]
        predict = [most_common] * len(eval_instances)
        score = []
        progress.start_task('Example', len(eval_instances))
        for i, inst in enumerate(eval_instances):
            progress.progress(i)
            score.append(np.log(self._get_smoothed_prob(inst.output)))
        progress.end_task()
        return predict, score

    def _get_smoothed_prob(self, output):
        if output in self.seen and self.seen[output] > 1:
            return (self.seen[output] - 1.0) / self.num_examples
        else:
            return 1.0 * len(self.seen) / self.num_examples


class UnigramLMSpeakerLearner(Learner):
    def __init__(self):
        options = config.options()
        self.tokenizer = options.speaker_tokenizer
        self.token_counts = Counter()
        self.seq_vec = SequenceVectorizer(unk_threshold=options.speaker_unk_threshold)
        self.num_tokens = 0

    def train(self, training_instances, validation_instances='ignored', metrics='ignored'):
        tokenize = TOKENIZERS[self.tokenizer]

        tokenized = [tokenize(inst.output) + ['</s>'] for inst in training_instances]
        self.seq_vec.add_all(tokenized)
        unk_replaced = self.seq_vec.unk_replace_all(tokenized)

        progress.start_task('Example', len(training_instances))
        for i, utt in enumerate(unk_replaced):
            progress.progress(i)
            self.token_counts.update(utt)
            self.num_tokens += len(utt)
        progress.end_task()

    @property
    def num_params(self):
        return len(self.token_counts)

    def predict_and_score(self, eval_instances):
        predict = [''] * len(eval_instances)
        score = []
        progress.start_task('Example', len(eval_instances))
        for i, inst in enumerate(eval_instances):
            progress.progress(i)
            score.append(self._get_log_prob(inst.output))
        progress.end_task()
        return predict, score

    def _get_log_prob(self, output):
        tokenize = TOKENIZERS[self.tokenizer]
        tokenized = tokenize(output) + ['</s>']
        unk_replaced = self.seq_vec.unk_replace(tokenized)
        log_prob = 0.0
        for token in unk_replaced:
            log_prob += np.log(self.token_counts[token] * 1.0 / self.num_tokens)
        return log_prob


class RandomListenerLearner(Learner):
    def train(self, training_instances, validation_instances='ignored', metrics='ignored'):
        self.num_params = 0

    def predict_and_score(self, eval_instances):
        predict = [(0 if isinstance(inst.output, Number) else (128, 128, 128))
                   for inst in eval_instances]
        score = [(-np.log(len(inst.alt_outputs))
                  if isinstance(inst.output, Number) else
                  -3.0 * np.log(256.0))
                 for inst in eval_instances]
        return predict, score


class LookupLearner(Learner):
    def __init__(self):
        options = config.options()
        self.counters = defaultdict(Counter)
        if options.listener:
            res = options.listener_color_resolution
            hsv = options.listener_hsv
        else:
            res = options.speaker_color_resolution
            hsv = options.speaker_hsv
        self.res = res
        self.hsv = hsv
        self.init_vectorizer()

    def init_vectorizer(self):
        if self.res and self.res[0]:
            if len(self.res) == 1:
                self.res = self.res * 3
            self.color_vec = BucketsVectorizer(self.res, hsv=self.hsv)
            self.vectorize = lambda c: self.color_vec.vectorize(c, hsv=True)
            self.unvectorize = lambda c: self.color_vec.unvectorize(c, hsv=True)
            self.score_adjustment = -np.log((256.0 ** 3) / self.color_vec.num_types)
        else:
            self.vectorize = lambda c: c
            self.unvectorize = lambda c: c
            self.score_adjustment = 0.0

    @property
    def num_params(self):
        return sum(len(c) for c in self.counters.values())

    def train(self, training_instances, validation_instances='ignored', metrics='ignored'):
        options = config.options()
        for inst in training_instances:
            inp, out = inst.input, inst.output
            if options.listener:
                out = self.vectorize(out)
            else:
                inp = self.vectorize(inp)
            self.counters[inp][out] += 1

    def predict_and_score(self, eval_instances, random='ignored', verbosity=0):
        options = config.options()
        if options.verbosity + verbosity >= 2:
            print('Testing')
        predictions = []
        scores = []
        for inst in eval_instances:
            inp, out = inst.input, inst.output
            if options.listener:
                out = self.vectorize(out)
            else:
                inp = self.vectorize(inp)

            counter = self.counters[inp]
            highest = counter.most_common(1)
            if highest:
                if options.listener:
                    prediction = self.unvectorize(highest[0][0])
                else:
                    prediction = highest[0][0]
            elif options.listener:
                prediction = (0, 0, 0)
            else:
                prediction = '<unk>'

            total = sum(counter.values())
            if total:
                if options.verbosity + verbosity >= 9:
                    print('%s -> %s: %s of %s [%s]' % (repr(inp), repr(out), counter[out],
                                                       total, inst.input))
                prob = counter[out] * 1.0 / total
            else:
                if options.verbosity + verbosity >= 9:
                    print('%s -> %s: no data [%s]' % (repr(inp), repr(out), inst.input))
                prob = 1.0 * (inst.output == prediction)
            score = np.log(prob)
            if options.listener:
                score += self.score_adjustment

            predictions.append(prediction)
            scores.append(score)

        return predictions, scores

    def __getstate__(self):
        return {
            'counters': {k: dict(v) for k, v in self.counters.iteritems()},
            'res': self.res,
            'hsv': self.hsv,
        }

    def __setstate__(self, state):
        self.res = state['res']
        self.hsv = state['hsv']
        self.init_vectorizer()
        self.counters = defaultdict(Counter, {k: Counter(v) for k, v in state['counters']})

class BaselineLearner(Learner):
    def __init__(self):
        self.model = LogisticRegression()

    # returns 1 if X is within x_eps around x's value in x_dict
    # returns -1 otherwise
    def indicator(self, X, x, x_dict, x_eps):
        return 1 if abs(X - x_dict[x]) <= x_eps else -1

    def make_features(self, instances):
        # dictionaries for representative values
        hue_dict = {'red' : 0, 'orange' : 30, 'yellow' : 60, 'green' : 120,
                    'cyan' : 180, 'blue': 240, 'purple': 270, 'magenta' : 300}
        sat_dict = {'dull' : 50, 'faded' : 50, 'pale' : 50, 'bright': 100}
        val_dict = {'dark' : 25, 'muted' : 50, 'light' : 75}

        # epsilon - the interval around the representative values
        hue_eps = 45
        sat_eps = 25
        val_eps = 25

        X = [[] for x in xrange(len(instances)*3)]

        for i, inst in enumerate(instances):
            inp, alt = inst.input, inst.alt_outputs
            # go through each color
            for j in xrange(3):
                H,S,V = alt[j][:]
                row = X[3*i+j]
                for w in self.top_words:
                    w_indicator = 1 if w in inp else -1

                    h_feats = [w_indicator * self.indicator(H,h,hue_dict,hue_eps)
                                for h in hue_dict.keys()]
                    s_feats = [w_indicator * self.indicator(S,s,sat_dict,sat_eps)
                                for s in sat_dict.keys()]
                    v_feats = [w_indicator * self.indicator(V,v,val_dict,val_eps)
                                for v in val_dict.keys()]

                    row += h_feats + s_feats + v_feats

                    # row.append(w_indicator*np.cos(H))
                    # row.append(w_indicator*np.sin(H))
        return X

    def top_words(self, instances):
        self.num_top_words = 100
        stops = set(stopwords.words("english"))
        with open('behavioralAnalysis/stopwords-en.txt') as f:
            additional_stops = f.readlines()
        stops = stops | set([x.strip() for x in additional_stops])
        all_words = []
        for inst in instances:
            all_words += map(lambda s : s.lower(), inst.input.split())
        self.top_words = [w for w, w_count in Counter(all_words).most_common(self.num_top_words)
                            if w not in stops and w.isalpha()]
        self.top_words[-1] = 'not'

    def train(self, training_instances, validation_instances='ignored', metrics='ignored'):
        self.num_params = 0 # change later

        print "finding top words..."
        self.top_words(training_instances)
        print "top %d words: " % self.num_top_words
        print self.top_words

        # make features for training dataset
        print "making features for training dataset..."
        self.X_train = self.make_features(training_instances)

        # transform outputs into one-hot vectors
        training_targets = np.zeros(3*len(training_instances))
        for i, inst in enumerate(training_instances):
            training_targets[3*i+inst.output] = 1

        # learn the parameters for the model
        print "training..."
        self.model.fit(self.X_train, training_targets)

    def predict_and_score(self, eval_instances):
        num_instances = len(eval_instances)

        # make features for eval dataset
        print "making features for eval dataset..."
        self.X_eval = self.make_features(eval_instances)

        # find log probabilities using model trained above
        print "finding probabilities..."
        log_probs = self.model.predict_log_proba(self.X_eval)[:,1]
        reshaped = np.reshape(log_probs,(num_instances,3))
        final_probs = reshaped - logsumexp(reshaped, axis=1, keepdims=True)

        preds = []
        scores = []
        print "making predictions..."
        progress.start_task('Example', len(eval_instances))
        for i, inst in enumerate(eval_instances):
            progress.progress(i)
            pred = np.argmax(final_probs[i])
            score = final_probs[i][inst.output]
            preds.append(pred)
            scores.append(score)
        progress.end_task()
        return preds, scores

class ChineseLearner(Learner):
    def __init__(self):
        self.model = LogisticRegression()
        # subcharacters to look for
        self.subchars = ['纟', '氵', '水', '火', '灬',
                        '艹', '木', '土', '日', '米', '女']
        # dictionaries for representative values
        self.hue_dict = {'红' : 0, '橙' : 30, '黄' : 60, '绿' : 120,
                        '海' : 180, '蓝' : 240, '紫' : 270, '粉' : 370}
        self.sat_dict = {'土' : 0, '灰' : 25, '淡' : 50, '亮' : 100}
        self.val_dict = {'墨' : 0, '深' : 25, '暗' : 25, '肝' : 25,
                        '淡' : 75, '浅' : 75}
        # epsilon - the interval around the representative values
        self.hue_eps = 55
        self.sat_eps = 30
        self.val_eps = 30

    # returns 1 if X is within x_eps around x's value in x_dict, else -1
    def in_range(self, X, x, attributeid):
        if attributeid == 'hue':
            d, eps = self.hue_dict, self.hue_eps
        elif attributeid == 'sat':
            d, eps = self.sat_dict, self.sat_eps
        elif attributeid == 'val':
            d, eps = self.val_dict, self.val_eps
        else:
            raise NameError('Invalid attributeid: try hue, sat, or val.')
        return 1 if abs(X - d[x]) <= eps else -1

    def is_subchar(self, char, subchar):
        import cjklib.characterlookup as cl
        cjk = cl.CharacterLookup('C')
        decomp = cjk.getDecompositionEntries(char)
        if decomp:
            subchars = decomp[0][1:]
            return subchar.decode('utf-8') in [x[0] for x in subchars]
        else:
            return False

    def negate(self, inp, row):
        if  u'不' in inp:
            split = inp.split(u'不')[1:]
            for s in split:
                for w in list(s):
                    if w in self.top_words:
                        k = self.top_words.index(w)
                        n = len(self.hue_dict.keys() + self.sat_dict.keys()
                                + self.val_dict.keys())
                        for l in xrange(k * n, k * n + n):
                            row[l] *= -1

    def subchar_feats(self, inp, H, row):
        for c in self.subchars:
            c_indicator = 1 if any([self.is_subchar(s, c)
                                    for s in inp]) else -1
            h_feats = [c_indicator * self.in_range(H, h, 'hue')
                        for h in self.hue_dict.keys()]
            row += h_feats

    def make_features(self, instances):
        X = [[] for x in xrange(len(instances) * 3)]
        for i, inst in enumerate(instances):
            inp, alt = inst.input, inst.alt_outputs
            closest = None
            # if u'最' in inp:
            #     s = inp.split(u'最')[1]
            #     for word in s: # each word in the substring after '最'
            #         w = word.encode('utf-8')
            #         if w in self.hue_dict.keys():
            #             closest = np.argmin([abs(alt[j][0] - self.hue_dict[w])
            #                                 for j in xrange(3)])
            #             break
            #         elif w in self.sat_dict.keys():
            #             closest = np.argmin([abs(alt[j][1] - self.sat_dict[w])
            #                                 for j in xrange(3)])
            #             break
            #         elif w in self.val_dict.keys():
            #             closest = np.argmin([abs(alt[j][2] - self.val_dict[w])
            #                                 for j in xrange(3)])
            #             break
            # go through each color
            for j in xrange(3):
                H, S, V = alt[j][:]
                row = X[3 * i + j]
                for w in self.top_words:
                #     if w in inp:
                #         if j == closest:
                #             w_indicator = 3
                #         else:
                #             w_indicator = 1
                #     else:
                #         w_indicator = -1
                    w_indicator = 1 if w in inp else -1
                    h_feats = [w_indicator * self.in_range(H, h, 'hue')
                                for h in self.hue_dict.keys()]
                    s_feats = [w_indicator * self.in_range(S, s, 'sat')
                                for s in self.sat_dict.keys()]
                    v_feats = [w_indicator * self.in_range(V, v, 'val')
                                for v in self.val_dict.keys()]
                    row += h_feats + s_feats + v_feats
                # check for negation
                self.negate(inp, row)
                # check for subchars and relationship with hue
                # self.subchar_feats(inp, H, row)
        # TODO: feature names
        return X

    def top_words(self, instances, num_top_words):
        with open('behavioralAnalysis/stopwords-zh.txt') as f:
            stops = f.readlines()
        stops = set([x.strip().decode('utf-8') for x in stops])
        inputs = [list(inst.input) for inst in instances]
        words = [w for inp in inputs for w in inp] # flatten
        ordered = [w for (w, w_count) in Counter(words).most_common()
                    if w.isalpha() and w not in stops]
        self.top_words = ordered[:num_top_words]

    def train(self, training_instances, validation_instances='ignored', metrics='ignored'):
        self.num_params = 0 # change later

        num_top_words = 70
        self.top_words(training_instances, num_top_words)
        print "top %d words: " % num_top_words
        print repr(self.top_words).decode('unicode_escape').encode('utf-8')

        # make features for training dataset
        print "making features for training dataset..."
        self.X_train = self.make_features(training_instances)

        # transform outputs into one-hot vectors
        training_targets = np.zeros(3 * len(training_instances))
        for i, inst in enumerate(training_instances):
            training_targets[3 * i + inst.output] = 1

        # learn the parameters for the model
        print "training..."
        self.model.fit(self.X_train, training_targets)

        weights = self.model.coef_


    def predict_and_score(self, eval_instances):
        num_instances = len(eval_instances)

        # make features for eval dataset
        print "making features for eval dataset..."
        self.X_eval = self.make_features(eval_instances)

        # find log probabilities using model trained above
        print "finding probabilities..."
        log_probs = self.model.predict_log_proba(self.X_eval)[:,1]
        reshaped = np.reshape(log_probs,(num_instances, 3))
        final_probs = reshaped - logsumexp(reshaped, axis=1, keepdims=True)

        preds, scores = [], []
        print "making predictions..."
        progress.start_task('Example', len(eval_instances))
        for i, inst in enumerate(eval_instances):
            progress.progress(i)
            pred = np.argmax(final_probs[i])
            score = final_probs[i][inst.output]
            preds.append(pred)
            scores.append(score)
        progress.end_task()
        return preds, scores

LEARNERS = {
    'Histogram': HistogramLearner,
    'Lux': LuxLearner,
    'RSA': RSALearner,
    'MostCommon': MostCommonSpeakerLearner,
    'UnigramLM': UnigramLMSpeakerLearner,
    'Random': RandomListenerLearner,
    'Lookup': LookupLearner,
    'Baseline': BaselineLearner,
    'Chinese': ChineseLearner
}
LEARNERS.update(LISTENERS)
LEARNERS.update(SPEAKERS)

# Break cyclic dependency: ExhaustiveS1Learner needs list of learners to define
# exhaustive_base_learner command line option, LEARNERS needs ExhaustiveS1Learner
# to be defined to include it in the list.
import ref_game
LEARNERS.update({
    'ExhaustiveS1': ref_game.ExhaustiveS1Learner,
    'ExhaustiveL2': ref_game.ExhaustiveL2Learner,
    'DirectRefGame': ref_game.DirectRefGameLearner,
    'LRContextListener': ref_game.LRContextListenerLearner,
})

import sampled_ams
LEARNERS.update({
    'ACGaussian': sampled_ams.ACGaussianLearner,
})
