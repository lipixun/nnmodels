# encoding=utf8

""" The main entry
    Author: lipixun
    Created Time : 2018-05-13 16:09:55

    File Name: main.py
    Description:

"""

import sys
reload(sys)
sys.setdefaultencoding("utf8")

import logging

from os import listdir
from os.path import join, isfile

from text import TextDictionary
from model import BiLSTMModel
from preproc import preprocess

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO)

if __name__ == "__main__":

    from argparse import ArgumentParser

    def get_args():
        """Get arguments
        """
        parser = ArgumentParser(description="ATEC NLP 20180418")
        sub_parsers = parser.add_subparsers(dest="action")

        preproc_parser = sub_parsers.add_parser("preproc", help="Preprocess")
        preproc_parser.add_argument("-i", "--input", dest="input", required=True, help="The input file or dir")
        preproc_parser.add_argument("-o", "--output", dest="output", required=True, help="The output file or dir")
        preproc_parser.add_argument("-d", "--dictionary-filename", dest="dictionary_filename", required=True, help="The dictionary filename")
        preproc_parser.add_argument("--no-fit", dest="no_fit", action="store_true", help="Do not fit dictionary")

        train_parser = sub_parsers.add_parser("train", help="Train")
        train_parser.add_argument("-n", "--name", dest="name", required=True, help="Model name")
        train_parser.add_argument("-d", "--dictionary-filename", dest="dictionary_filename", required=True, help="The dictionary filename")
        train_parser.add_argument("--train-input", dest="train_input", required=True, help="The train input file")
        train_parser.add_argument("--test-input", dest="test_input", help="The test input file")

        predict_parser = sub_parsers.add_parser("predict", help="Predict")
        predict_parser.add_argument("-n", "--name", dest="name", required=True, help="Model name")
        predict_parser.add_argument("-d", "--dictionary-filename", dest="dictionary_filename", required=True, help="The dictionary filename")
        predict_parser.add_argument("-i", "--input", dest="input", required=True, help="The input file or dir")
        predict_parser.add_argument("--with-score", dest="with_score", action="store_true", help="Print result with score")

        return parser.parse_args()

    def main():
        """The main entry
        """
        args = get_args()

        if args.action == "preproc":
            preprocess(args.input, args.output, args.dictionary_filename, args.no_fit)
        elif args.action == "train":
            # Load dictionary
            text_dict = TextDictionary.load(args.dictionary_filename)
            m = BiLSTMModel(args.name, text_dict.id_size)
            filenames = [join(args.train_input, x) for x in listdir(args.train_input) if isfile(join(args.train_input, x))]
            m.train(filenames, None)
        elif args.action == "predict":
            text_dict = TextDictionary.load(args.dictionary_filename)
            m = BiLSTMModel(args.name, text_dict.id_size)
            filenames = [join(args.input, x) for x in listdir(args.input) if isfile(join(args.input, x))]
            results = m.predict(filenames)
            for line_no, score in results:
                if args.with_score:
                    print "%s\t%d\t%.4f" % (line_no, 1 if score >= 0.5 else 0, score)
                else:
                    print "%s\t%d" % (line_no, 1 if score >= 0.5 else 0)
        else:
            raise ValueError("Unknown action [%s]" % args.action)

    main()
