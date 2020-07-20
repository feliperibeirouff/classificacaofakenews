from logger import config_log, print2
from datetime import datetime
import csv

def get_dataset_params(version):
    if version == 'v1':
        params = {
            'train_base': 'features100_train.tsv',
            'validate_base': 'features100_validate.tsv',
            'min_words': 6,
            'max_rows_train': 10000,
            'max_rows_validate': 10000
        }
    elif version == 'v1.1':
        params = {
            'train_base': 'features100_train.tsv',
            'validate_base': 'features100_validate.tsv',
            'min_words': 6,
            'max_rows_train': 10000,
            'max_rows_validate': 3000
        }
    elif version == 'v2':
        params = {
            'train_base': 'features100_train.tsv',
            'validate_base': 'features100_validate.tsv',
            'test_base': 'features100_test.tsv',
            'min_words': 6
        }
    elif version == 'v3':
        params = {
            'train_base': 'features300_train.tsv',
            'validate_base': 'features300_validate.tsv',
            'test_base': 'features300_test.tsv',
            'min_words': 6
        }
    return params

class Reader:
    def __init__(self, input_path, version):
        self.input_path = input_path
        self.dataset_params = get_dataset_params(version)

    def read_train(self):
        return self.read_features(self.dataset_params['train_base'], self.dataset_params.get('max_rows_train'))

    def read_validate(self):
        return self.read_features(self.dataset_params['validate_base'], self.dataset_params.get('max_rows_validate'))

    def read_test(self):
        return self.read_features(self.dataset_params['test_base'], self.dataset_params.get('max_rows_test'))

    def read_features(self, filename, max_rows):
        tsv_file = open(self.input_path + filename, encoding="utf8")
        read_tsv = csv.reader(tsv_file, delimiter="\t")

        X = []
        y = []
        i = 0
        min_words = self.dataset_params.get('min_words')

        print2('max_rows:', max_rows, 'min_words', min_words, 'filename', filename, str(datetime.now()))

        for row in read_tsv:
            if i == 0:
                i = 1
                continue
            num_words = int(row[2])
            if min_words is None or num_words >= min_words:
                i = i + 1
                way_label2 = row[3]
                y.append(way_label2)
                list_of_floats = [float(item) for item in row[6].split(' ')]
                X.append(list_of_floats)
            if i % 10000 == 0:
                print(i)
            if max_rows != None and i > max_rows:
                break

            #    break
                #print2(X)
                #print2(y)

        return [X, y]