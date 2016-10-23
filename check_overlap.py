from collections import Counter
import color_instances as ci


def check_unique_ids(name, insts):
    ids = Counter(inst.source for inst in insts)
    if len(ids) != len(insts):
        print('Duplicate source ids in %s: %s/%s %s' %
              (name, len(insts) - len(ids), len(insts), ids.most_common(1)[0]))
    return name, insts


def check_nonoverlapping(a_name, a, b_name, b):
    a_ids = set(inst.source for inst in a)
    b_ids = set(inst.source for inst in b)

    num_overlapping = len(a_ids.intersection(b_ids))
    if num_overlapping:
        print('Overlap %s vs %s: %s/%s/%s (%.1f%%/%.1f%%)' %
              (a_name, b_name, num_overlapping, len(a), len(b),
               num_overlapping * 100.0 / len(a), num_overlapping * 100.0 / len(b)))


if __name__ == '__main__':
    train = check_unique_ids('train', ci.hawkins_train(listener=True))
    dev = check_unique_ids('dev', ci.hawkins_dev(listener=True))
    test = check_unique_ids('test', ci.hawkins_test(listener=True))
    tune_train = check_unique_ids('tune_train', ci.hawkins_tune_train(listener=True))
    tune_test = check_unique_ids('tune_test', ci.hawkins_tune_test(listener=True))

    big_train = check_unique_ids('big_train', ci.hawkins_big_train(listener=True))
    big_dev = check_unique_ids('big_dev', ci.hawkins_big_dev(listener=True))
    big_test = check_unique_ids('big_test', ci.hawkins_big_test(listener=True))
    big_tune_train = check_unique_ids('big_tune_train', ci.hawkins_big_tune_train(listener=True))
    big_tune_test = check_unique_ids('big_tune_test', ci.hawkins_big_tune_test(listener=True))

    nonoverlapping_pairs = [
        # Ordinary guarantee: non-overlapping splits
        (train, dev),
        (train, test),
        (dev, test),
        (big_train, big_dev),
        (big_train, big_test),
        (big_dev, big_test),

        # Tuning should also be independent
        (tune_train, tune_test),
        (big_tune_train, big_tune_test),
        # And independent of both dev and test
        (tune_test, dev),
        (tune_test, test),
        (big_tune_test, big_dev),
        (big_tune_test, big_test),

        # Shouldn't be evaluating on dev or train of smaller dataset
        (train, big_dev),
        (train, big_test),
        (dev, big_dev),
        (dev, big_test),
    ]

    for (a_name, a), (b_name, b) in nonoverlapping_pairs:
        check_nonoverlapping(a_name, a, b_name, b)
