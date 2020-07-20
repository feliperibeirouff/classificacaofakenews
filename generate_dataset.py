import numpy as np
import csv
import pandas
import fasttext.util

def array_to_str(l, separator=" "):
    return separator.join(np.char.mod('%f', l))

import io

#Substitua aqui pelo seu diretório local:
input_path = "H:\\mestrado\\aprendizadomaquina\\trabalho2\\datasetfakenews\\"

fasttext.util.download_model('en', if_exists='ignore')
#ft = fasttext.load_model('cc.en.100.bin')
ft = fasttext.load_model('cc.en.300.bin')
fasttext.util.reduce_model(ft, 100)
ft.save_model('cc.en.100.bin')

save = False

def generate_featues(filename, features_filename):
    tsv_file = open(input_path + filename, encoding="utf8")
    read_tsv = csv.reader(tsv_file, delimiter="\t")
    writer = open(input_path + features_filename, "w")

    corretas = 0
    incorretas = 0

    i = 0
    for row in read_tsv:
        i = i + 1
        clean_title = row[5]
        title = row[15]
        numWords = len(clean_title.split(" "))
        hasImagem = row[8]
        id = row[0]
        way_label2 = row[17]
        way_label3 = row[18]
        way_label6 = row[19]
        if (numWords == 2):
            #print(clean_title)
            if way_label2 == '0':
                corretas = corretas+1
            if way_label2 == '1':
                incorretas = incorretas + 1

            print(numWords, way_label2, clean_title)
            writer.write(row[0] + "\t" + clean_title + "\t" + str(numWords) + "\t" + way_label2 + "\t" + way_label3 + "\t" + way_label6 + "\t" + array_to_str(ft.get_sentence_vector(clean_title)) + "\n")
            #if i % 1000 == 0:
            #    print(i)
    print(numWords, corretas, incorretas, corretas/(corretas+incorretas), incorretas/(corretas+incorretas))
    writer.close()

generate_featues("train.tsv", "features100_train.tsv")
generate_featues("validate.tsv", "features100_validate.tsv")
generate_featues("test.tsv", "features100_test.tsv")