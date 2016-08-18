'''
Utility functions for working with Instances, when speaker and listener instances
are floating around and it can be hard to keep track of when to use the "input"
and when to use the "output". Each method takes an instance and a flag to
indicate whether the instance is a speaker (input=color, output=utterance) or
listener (input=utterance, output=color) instance.
'''
from numbers import Number

from stanza.research import instance


def get_utt_simple(inst, listener):
    return inst.input if listener else inst.output


def get_utt(inst, listener):
    return get_multi(get_utt_simple(inst, listener))


def get_color_index(inst, listener):
    return inst.output if listener else inst.input


def get_color(inst, listener):
    index = get_color_index(inst, listener)
    if isinstance(index, Number):
        return inst.alt_outputs[index] if listener else inst.alt_inputs[index]
    else:
        return index


def get_context(inst, listener):
    return inst.alt_outputs if listener else inst.alt_inputs


def build_instance(utt, target, context, listener):
    if listener:
        return instance.Instance(utt, target, alt_outputs=context)
    else:
        return instance.Instance(target, utt, alt_inputs=context)


def get_speaker_inst(inst, listener):
    if listener:
        return instance.inverted()
    else:
        return instance


def get_multi(val):
    if isinstance(val, tuple):
        assert len(val) == 1
        return val[0]
    else:
        return val
