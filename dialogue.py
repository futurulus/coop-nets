import numpy as np
import theano.tensor as T
from lasagne.layers import InputLayer, DenseLayer
from lasagne.objectives import categorical_crossentropy
from lasagne.nonlinearities import softmax

from stanza.monitoring import progress
from stanza.research import iterators, learner
from neural import NeuralLearner, SimpleLasagneModel, OPTIMIZERS, sample
import color_instances as ci


class ReprNextActionLearner(NeuralLearner):
    def __init__(self, base=None, id=None):
        super(ReprNextActionLearner, self).__init__(id=id)
        options = config.options()
        if base is None:
            self.base = learners.new(options.repr_learner)
            with open(options.repr_model, 'rb') as infile:
                self.base.load(infile)
        else:
            self.base = base

        # all vectorization is delegated to base model
        self.seq_vec = self.color_vec = None

    def predict_and_score(self, eval_instances, random=False, verbosity=0):
        predictions = []
        scores = []

        batches = iterators.iter_batches(eval_instances, self.options.listener_eval_batch_size)
        num_batches = (len(eval_instances) - 1) // self.options.listener_eval_batch_size + 1

        if self.options.verbosity + verbosity >= 2:
            print('Testing')
        progress.start_task('Eval batch', num_batches)
        for batch_num, batch in enumerate(batches):
            progress.progress(batch_num)
            batch = list(batch)

            xs, (y,) = self._data_to_arrays(batch, test=True)

            probs = self.model.predict(xs)
            if random:
                indices = sample(probs)
                predictions.extend(indices)
            else:
                predictions.extend(probs.argmax(axis=1))
            scores_arr = np.log(probs[np.arange(len(batch)), y])
            scores.extend(scores_arr.tolist())
        progress.end_task()
        if self.options.verbosity >= 9:
            print('%s %ss:') % (self.id, 'sample' if random else 'prediction')
            for inst, prediction in zip(eval_instances, predictions):
                print('%s -> %s' % (repr(inst.input), repr(prediction)))

        return predictions, scores

    def train_priors(self, training_instances, listener_data=False):
        prior_class = listener.PRIORS[self.options.listener_prior]
        self.prior_emp = prior_class()
        self.prior_smooth = prior_class()

        self.prior_emp.train(training_instances, listener_data=listener_data)
        self.prior_smooth.train(training_instances, listener_data=listener_data)

    def _data_to_arrays(self, training_instances,
                        init_vectorizer=False, test=False, inverted=False):
        utts = []
        ys = []

        for inst in training_instances:
            utts.append(inst.input)
            ys.append(inst.output)

        reprs = self.base.get_reprs(utts)
        assert np.isfinite(reprs).all()
        return [reprs], [np.array(ys, dtype=np.int32)]

    def _build_model(self, model_class=SimpleLasagneModel):
        self.get_options()
        id_tag = (self.id + '/') if self.id else ''

        input_var = T.matrix(id_tag + 'inputs')
        target_var = T.ivector(id_tag + 'targets')

        self.l_out, self.input_layers = self._get_l_out([input_var])
        self.loss = categorical_crossentropy

        self.model = model_class(
            [input_var], [target_var], self.l_out,
            loss=self.loss, optimizer=OPTIMIZERS[self.options.listener_optimizer],
            learning_rate=self.options.listener_learning_rate,
            id=self.id)

    def _get_l_out(self, input_vars):
        id_tag = (self.id + '/') if self.id else ''

        repr_size = self.base.get_reprs(['red']).shape[1]

        input_var = input_vars[0]
        l_in = InputLayer(shape=(None, repr_size), input_var=input_var,
                          name=id_tag + 'input_repr')

        l_out = DenseLayer(l_in, num_units=3, nonlinearity=softmax,
                           name=id_tag + 'output_pred')

        return l_out, [l_in]


class BaselineNextActionLearner(learner.Learner):
    def train(self, *args, **kwargs):
        pass

    @property
    def num_params(self):
        return 0

    def predict_and_score(self, eval_instances, random=False, verbosity=0):
        predictions = []
        scores = []

        progress.start_task('Instance', len(eval_instances))
        for inst_num, inst in enumerate(eval_instances):
            progress.progress(inst_num)

            pred, score = self.predict_one_inst(inst)
            predictions.append(pred)
            scores.append(score)
        progress.end_task()

        return predictions, scores

    def predict_one_inst(self, inst):
        lines = inst.input.split(' ~ ')
        last_line = lines[-1]

        # 1. if last line is listener utt, predict NONE: *50%, SPEAK: 30%, CHOOSE: 20%
        if last_line.startswith('| '):
            return ci.ACTION_NONE, np.log([0.5, 0.3, 0.2][inst.output])
        # 2. if no color term yet, predict NONE: 40%, SPEAK: *40%, CHOOSE: 20%
        elif not has_color_word(' '.join(line for line in lines if not line.startswith('| '))):
            return ci.ACTION_SPEAK, np.log([0.4, 0.4, 0.2][inst.output])
        # 3. otherwise predict NONE: 12%, SPEAK: 8%, CHOOSE: *80%
        else:
            return ci.ACTION_CHOOSE, np.log([0.12, 0.08, 0.8][inst.output])


def has_color_word(speaker_utts):
    '''
    >>> has_color_word('the blue one')
    True
    >>> has_color_word('reddish I think')
    True
    >>> has_color_word('great job')
    False
    '''
    from nltk.corpus import wordnet as wn
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    wnl = WordNetLemmatizer()
    words = [wnl.lemmatize(word) for word in set(word_tokenize(speaker_utts))]
    for word in words:
        nounForms = wn.synsets(word, pos='n')
        nounSynsets = nounForms if nounForms else nounify(word)
        if any(('color.n.01' in [s.name() for s in flatten(n.hypernym_paths())])
               for n in nounSynsets):
            return True
    return False


def flatten(l):
    return [item for sublist in l for item in sublist]


def nounify(adj_word):
    from nltk.corpus import wordnet as wn
    """ Transform an adjective to the closest noun: dead -> death """
    adj_synsets = wn.synsets(adj_word, pos="a")

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


from stanza.research import config
import learners
import listener

parser = config.get_options_parser()
parser.add_argument('--repr_learner', default='Speaker',
                    choices=learners.LEARNERS.keys(),
                    help='The name of the model to use to get representations for classifying '
                         'the next action to take after an utterance.')
parser.add_argument('--repr_model', default=None,
                    help='The path to the model to use to get representations for classifying '
                         'the next action to take after an utterance.')
