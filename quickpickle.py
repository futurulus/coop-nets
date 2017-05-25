import cPickle as pickle
import types
import sys

import run_experiment  # NOQA: make sure we load all the command line args
from stanza.research import config


def patch(model):
    def __quickpickle_setstate__(self, state):
        self.__dict__ = state

    def __quickpickle_getstate__(self):
        state = dict(self.__dict__)
        del state['__getstate__']
        del state['__setstate__']
        state['quickpickle'] = True
        return state

    def __quickpickle_numparams__(self):
        return self.quickpickle_numparams

    model.__getstate__ = types.MethodType(__quickpickle_getstate__, model)
    model.__setstate__ = types.MethodType(__quickpickle_setstate__, model)
    model.quickpickle_numparams = model.num_params


if __name__ == '__main__':
    sys.setrecursionlimit(50000)
    options = config.options(read=True)
    if options.load:
        modelfile = options.load
    else:
        modelfile = config.get_file_path('model.p')
    with open(modelfile, 'rb') as infile, config.open('quickpickle.p', 'wb') as outfile:
        model = pickle.load(infile)
        patch(model)
        pickle.dump(model, outfile)
