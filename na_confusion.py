#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Usage:
#   python na_confusion.py --run_dir runs/l2
# run dir should have data.eval.jsons.
import os
import json
from sklearn.metrics import classification_report

from stanza.research import config


def print_confusion_matrix():
    options = config.options(read=True)

    data_path = os.path.join(options.run_dir, 'data.eval.jsons')
    with open(data_path, 'r') as infile:
        gold = [json.loads(line.strip())['output'] for line in infile]

    preds_path = os.path.join(options.run_dir, 'predictions.eval.jsons')
    with open(preds_path, 'r') as infile:
        preds = [json.loads(line.strip()) for line in infile]

    print classification_report(gold, preds,
                                target_names=['none', 'speak', 'choose'])



if __name__ == '__main__':
    print_confusion_matrix()
