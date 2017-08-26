from utils import *
import argparse

def compare(attribute, generate_en=False):
    '''
    Performs a full comparison across Chinese and English for a given
    attribute. This includes generating the csv files and plotting the data.
    '''
    print 'Comparing {}...'.format(attribute)
    generate_csv(attribute, 'zh')
    if generate_en:
        generate_csv(attribute, 'en')
    plot_for_attribute(attribute)

if __name__ == '__main__':
    ATTRIBUTES = ['tokens', 'dialogue', 'superlative', 'comparative',
                  'negation', 'specificity', 'success']
    parser = argparse.ArgumentParser(description='compare Chinese and English')
    parser.add_argument('--mode', help='compare, generate, or plot',
                        default=None, choices=['compare', 'csv', 'plot'])
    parser.add_argument('--attributes', help='choose attributes',
                        nargs='*', type=str, default=None,
                        choices=ATTRIBUTES+['all'])
    parser.add_argument('--language', help='choose language',
                        default=None, choices=['zh', 'en'])
    parser.add_argument('--plot_type', help='choose type of plot',
                        default='bar', choices=['bar', 'box', 'hist']) # TODO
    args = parser.parse_args()

    attrs = ATTRIBUTES if args.attributes == ['all'] else args.attributes
    if args.mode == 'compare':
        for a in attrs:
            print
            compare(a, args.language == 'en')
    elif args.mode == 'csv':
        for a in attrs:
            generate_csv(a, args.language)
    elif args.mode == 'plot':
        for a in attrs:
            plot_for_attribute(a)
    print 'Done.'
