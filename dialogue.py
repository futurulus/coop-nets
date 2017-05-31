import numpy as np
import theano.tensor as T
from lasagne.layers import InputLayer, DenseLayer
from lasagne.objectives import categorical_crossentropy
from lasagne.nonlinearities import softmax

from stanza.monitoring import progress
from stanza.research import iterators
from neural import NeuralLearner, SimpleLasagneModel, OPTIMIZERS, sample


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
