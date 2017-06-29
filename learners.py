from collections import defaultdict, Counter
from numbers import Number
import numpy as np

from sklearn.linear_model import LogisticRegression
from scipy.misc import logsumexp

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


'''
JENN'S BASELINE LEARNER 06/28/2017
'''

class JennsLearner(Learner):

    # # saturation words boundary=75 on a 0-100 saturation scale in HSV
    # saturation_words = (
    #     ['pure', 'solid', 'rich', 'strong', 'harsh', 'intense'],
    #     ['white', 'gray','grey', 'faded', 'pale' 'bleached', 'pastel', 'mellow', 'muted', 'baby', 'dull']
    # )

    # # value words boundary=50 on 0-100 value scale in HSV
    # value_words = ('dark', 'deep', 'muted' 
    #                 'light', 'bright')

    def __init__(self):
        options = config.options()
        self.res = [2] #self.res = options.listener_color_resolution
        self.hsv = options.listener_hsv
        
        # sklearn model
        self.model = LogisticRegression()
        
    def vectorize_all(self, c):
        color_vec = FourierVectorizer(self.res, hsv=self.hsv)
        return color_vec.vectorize_all(c)

    def make_features(self, instances):
        color_words = ('red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'purple', 'magenta')
        
        hsv_dict = {
            'red' : 0,
            'orange' : 30,
            'yellow' : 60,
            'green' : 120,
            'cyan' : 180,
            'blue': 240,
            'purple': 270,
            'magenta' : 300
        }

        # constants
        hue_interval = 45
        num_instances = len(instances)
        num_features = len(color_words) # TEMPORARY!!!!

        # initialize to zeros
        X = np.zeros((num_instances*3, num_features))

        for i, inst in enumerate(instances):
            # get input (utterance) and alt outputs (colors)
            inp, alt = inst.input, inst.alt_outputs

            # fourier vectorize
            # fourier_colors = FourierVectorizer(self.res, hsv=self.hsv).vectorize_all(alt)
            # fourier_colors = self.vectorize_all(alt)
            
            # go through each color
            for j in xrange(3):
                c_ij = alt[j]

                for k, c in enumerate(color_words):
                    hue_indicator = 1 if abs(c_ij[0] - hsv_dict[c]) <= hue_interval else -1
                    word_indicator = 1 if c in inp else -1

                    # if i <=5:

                    #     print "COLOR: ", c
                    #     print "hue of c_ij: ", c_ij[0]
                    #     print "hue of %s: %d" % (c, hsv_colors[c])
                    #     print "diff: %d" % (c_ij[0] - hsv_colors[c])
                    #     print "color word present: ", color_word_dict[c]
                    #     print "INDICATOR: ", color_word_dict[c] * hue_indicator
                    #     print

                    X[i*3 + j][k] = word_indicator * hue_indicator

        return X

    def train(self, training_instances, validation_instances='ignored', metrics='ignored'):
        self.num_params = 0 # WHAT IS THIS???

        self.X_train = self.make_features(training_instances)
        print "X_train: ", self.X_train

        # transform outputs into ``one-hot coded'' vectors
        training_targets = np.zeros(3*len(training_instances)) # initialize to all zeros
        for i, inst in enumerate(training_instances):
            out = inst.output
            training_targets[3*i + out] = 1

        # learn the parameters for the model
        self.model.fit(self.X_train, training_targets)

    def predict_and_score(self, eval_instances):
        num_instances = len(eval_instances)

        # make features for eval dataset
        self.X_eval = self.make_features(eval_instances)

        # find log probabilities using model trained above
        log_probs = self.model.predict_log_proba(self.X_eval)

        # only keep probabilities associated with class 1
        class_one_log_probs = np.delete(log_probs, 0, axis=1)
    
        # reshape to allow for vectorized operations
        reshaped = np.reshape(class_one_log_probs,(num_instances,3))

        # use softmax to create probability distribution
        final_probs = reshaped - logsumexp(reshaped)

        preds = []
        scores = []
        progress.start_task('Example', len(eval_instances))
        for i, inst in enumerate(eval_instances):
            progress.progress(i)

            pred = np.argmax(final_probs[i])
            score = final_probs[i][inst.output]
            preds.append(pred)
            scores.append(np.log(score))

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
    'Jenns' : JennsLearner
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
