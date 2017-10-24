#!/usr/bin/env python
"""
Cleans up the data from the chinese message corpus
1 - Removes messages greater than 4 standard deviations away from the mean
2 - Removes spam games
Author: Andrew
"""
import pandas as pd

# === Constants ===
IN_DIR = 'data_input_raw/'
OUT_DIR = 'data_input_cleaned/'
INPUT_FILE = 'colorReferenceMessageChinese.csv'
MESSAGE_COLUMN = 'contents'
SEPARATOR = ','
NUM_COLUMNS = 5 # num columns in the csv files

# === Parameters ===
MAX_STD = 4 # for length
MAX_DUPLICATES = 25 # for spam. this number is the maximum number of the same reoccuring message in a game

def main():
    """
    Performs preprocessing (cleanup) on the chinese message data
    """
    # remove extra commas from the messages
    stripped_file_name = remove_extra_separators(IN_DIR + INPUT_FILE)
    opened_stripped_f = open(stripped_file_name, 'r')

    # read csv into a pandas df
    raw_df = pd.read_csv(opened_stripped_f, sep=SEPARATOR)
    print 'Columns: ', raw_df.columns
    raw_num_rows = raw_df.shape[0]
    print 'Original num datapoints: ', raw_num_rows

    # filter out the long messages and get information
    print 'Filtering out messages with lengths exceeding {} standard deviations.'.format(MAX_STD)
    filtered_len_df, deleted_len_df, mean, std = filter_long_messages(raw_df, MAX_STD)
    fil_len_num_rows = filtered_len_df.shape[0]
    # print summary results for filtered by length
    print 'Mean: {}, std: {}'.format(mean, std)
    print 'Num messages removed by excessive length:', raw_num_rows - fil_len_num_rows
    print 'Num datapoints remaining:', fil_len_num_rows

    # filter out the spam games
    print '\nFiltering out messages with spams containing more than {} duplicates.'.format(MAX_DUPLICATES)
    filtered_spam_df, deleted_spam_df = filter_spam_games(filtered_len_df, MAX_DUPLICATES)
    num_spam_games = len(deleted_spam_df.groupby('gameid'))
    fil_spam_num_rows = filtered_spam_df.shape[0]
    # print summary results for filtered by spam
    print 'Num spam games: ', num_spam_games
    print 'Num messages removed by spam:', fil_len_num_rows - fil_spam_num_rows
    print 'Num datapoints remaining:', fil_spam_num_rows

    # Write filtered to file
    print 'Writing to file'
    filtered_out_name = OUT_DIR + INPUT_FILE[:-4] + '_filtered.csv'
    print 'Writing filtered to ', filtered_out_name
    filtered_spam_df.to_csv(filtered_out_name, encoding='utf-8')

    # Write what got removed, to check what got removed makes sense
    del_len_out_name = OUT_DIR + 'filtered_deleted_long_lengths.csv'
    print 'Writing removed messages to ', del_len_out_name
    deleted_len_df.to_csv(del_len_out_name, encoding='utf-8')

    del_spam_out_name = OUT_DIR + 'filtered_deleted_spam.csv'
    print 'Writing removed messages to ', del_spam_out_name
    deleted_spam_df.to_csv(del_spam_out_name, encoding='utf-8')

    print 'Done.'


def remove_extra_separators(input_file):
    """
    This function is kind of pre-preprocessing. The chinese messages corpus is only supposed to have 5 columns per 
    datapoint. Any extra separators that occur in the dialog needs to be removed for the file to be read correctly. 
    The number of columns is set with the constant NUM_COLUMNS.

    :param input_file: the raw converted csv file from jsonToCSV.py
    :returns: name of the file stripped of extra commas
    """
    f = open(input_file, 'r')
    out_f = open('stripped_extra_commas.csv', 'w')
    for line in f:
        split = line.split(SEPARATOR)
        if len(split) > NUM_COLUMNS:
            # gameid, epochTime, roundNum, role, message,could,have,extra,separators
            num_seps = NUM_COLUMNS - 1
            # rejoin the previous columns with the original separator, then fill in spaces on the extraneous seps
            new_line = SEPARATOR.join(split[slice(num_seps)]) + ' '.join(split[slice(num_seps, len(split))])
        else:
            new_line = SEPARATOR.join(split)
        out_f.write(new_line)

    f.close()
    return out_f.name

def filter_long_messages(raw_df, std=4):
    """
    Removes messages with lengths GREATER than the number of standard deviations given.

    :param df: pandas dataframe
    :param std: number of standard deviations

    :returns: the filtered df, the rows that were removed, the mean, the std
    """
    # We have to use msg.decode('utf-8') so that chinese characters are properly evaluated as 1 character.
    # Technically this means that messages with english will be more likely to be thrown out, as spaces and letters
    # are single characters. This may or may not be an issue.
    raw_df['raw_lengths'] = raw_df[MESSAGE_COLUMN].astype(str).apply(lambda msg: len(msg.decode('utf-8')))
    length_mean = raw_df['raw_lengths'].mean()
    length_std = raw_df['raw_lengths'].std()
    raw_df['z_scores'] = (raw_df['raw_lengths'] - length_mean) / length_std
    # keep based on z_score (the num of standard deviations)
    filtered_length_df = raw_df[raw_df['z_scores'] <= std]
    deleted_length_df = raw_df[raw_df['z_scores'] > std]

    # remove the columns we added for calculating and filtering
    filtered_length_df = filtered_length_df.drop('raw_lengths', axis=1)
    filtered_length_df = filtered_length_df.drop('z_scores', axis=1)

    return filtered_length_df, deleted_length_df, length_mean, length_std

def filter_spam_games(df, threshold):
    """
    Filter out games (based on gameid) with duplicated messages greater than or equal to the threshold value. 
    A message is considered to be a duplicate if it appears more than once.
    E.g. if the threshold is 5, games with 5 or more duplicated messages

    :param df: the message df to filter
    :param threshold: the max number (exclusive) of duplicate messages. The max duplicated message is compared to the 
        threshold, ie 3 occurences of 'aaa' and 2 occurences of 'bbb' has a max duplicated value of 3, thus 3 is used.
    :returns: tuple first containing the filtered dataframe, and second the rows that were deleted
    """
    # filter out bad games. group by id, then group by the contents, then get the max size of the grouped contents
    filtered_spam_df = df.groupby('gameid').filter(lambda g: g.groupby('contents').size().max() < threshold)
    deleted_spam_df = df.groupby('gameid').filter(lambda g: g.groupby('contents').size().max() >= threshold)

    return filtered_spam_df, deleted_spam_df


if __name__ == '__main__':
    main()

