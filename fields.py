from stanza.research import instance


def get_utt(inst, listener):
    return inst.input if listener else inst.output


def get_color_index(inst, listener):
    return inst.output if listener else inst.input


def get_context(inst, listener):
    return inst.alt_outputs if listener else inst.alt_inputs


def build_instance(utt, target, context, listener):
    if listener:
        return instance.Instance(utt, target, alt_outputs=context)
    else:
        return instance.Instance(target, utt, alt_inputs=context)
