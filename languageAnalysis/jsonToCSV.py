import json
import sqlite3
import csv
import codecs
import cStringIO

DIR='data_input_raw/'
CHAT_IN_FILE = DIR+'chatmessages.json'
CHAT_OUT_FILE = DIR+'colorReferenceMessageChinese.csv'
CLICKED_IN_FILE = DIR+'clickedobjs.json'
CLICKED_OUT_FILE = DIR+'colorReferenceClicksChinese.csv'
SEPARATOR=','

class UnicodeWriter:
    """
    A CSV writer which will write rows to CSV file "f",
    which is encoded in the given encoding.
    """

    def __init__(self, f, dialect=csv.excel, encoding="utf-8", **kwds):
        # Redirect output to a queue
        self.queue = cStringIO.StringIO()
        self.writer = csv.writer(self.queue, dialect=dialect, delimiter=SEPARATOR, **kwds)
        self.stream = f
        self.encoder = codecs.getincrementalencoder(encoding)()

    def writerow(self, row):
        self.writer.writerow([s.encode("utf-8") for s in row])
        # Fetch UTF-8 output from the queue ...
        data = self.queue.getvalue()
        data = data.decode("utf-8")
        # ... and reencode it into the target encoding
        data = self.encoder.encode(data)
        # write to the target stream
        self.stream.write(data)
        # empty queue
        self.queue.truncate(0)

    def writerows(self, rows):
        for row in rows:
            self.writerow(row)

def COLORSjsonToCSV(header, infile, outfile, strip):
    with open(infile) as json_data:
        json_objs = json_data.readlines()
        with open(outfile, 'wb') as out:
            print 'Writing to ', outfile
            csvOut = UnicodeWriter(out)
            csvOut.writerow(header)
            for obj in json_objs:
                data = json.loads(obj)
                data_as_str = data['line'].strip()
                # about 14 datapoints in the json have random 'undefined' entries 
                data_list = filter(lambda x : x != 'undefined',
                            [x.strip() for x in data_as_str.split(SEPARATOR)])
                if strip:
                    data_list[-1] = data_list[-1][1:-1]
                csvOut.writerow(data_list)

if __name__ == '__main__':
    chatHeader = ['gameid','epochTime','roundNum','sender','contents']
    clickedHeader = ['gameid','time','roundNum','condition','clickStatus',
                    'clickColH','clickColS','clickColL','clickLocS',
                    'clickLocL','alt1Status','alt1ColH','alt1ColS','alt1ColL',
                    'alt1LocS','alt1LocL','alt2Status','alt2ColH','alt2ColS',
                    'alt2ColL','alt2LocS','alt2LocL','targetD1Diff',
                    'targetD2Diff','D1D2Diff','outcome']

    COLORSjsonToCSV(chatHeader, CHAT_IN_FILE, CHAT_OUT_FILE, True)
    COLORSjsonToCSV(clickedHeader, CLICKED_IN_FILE, CLICKED_OUT_FILE, False)
