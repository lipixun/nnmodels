# encoding=utf8

""" The main entry
    Author: lipixun
    Created Time : 2018-05-13 16:09:55

    File Name: main.py
    Description:

"""

from __future__ import print_function

import six
import sys

if six.PY2:
    reload(sys)
    sys.setdefaultencoding("utf8")

import logging

import tftrainer

from text import TextDictionary
from model import BiLSTMModel
from preproc import Preprocessor

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO)

def preproc(input_path, output_path, dict_path, use_jieba, no_fit):
    """Preprocess
    """
    preprocessor = Preprocessor(use_jieba)
    preprocessor.preprocess(input_path, output_path, dict_path, no_fit)

def train(name, attention, workpath, dict_file, train_files, eval_files, epoch):
    """Train the model
    """
    train_files = tftrainer.path.findfiles(train_files)
    eval_files = tftrainer.path.findfiles(eval_files)
    if not train_files:
        raise ValueError("Require train files")

    print("Train files:", file=sys.stderr)
    for filename in train_files:
        print("\t%s" % filename, file=sys.stderr)

    print("Evaluate files:", file=sys.stderr)
    for filename in eval_files:
        print("\t%s" % filename, file=sys.stderr)

    text_dict = TextDictionary()
    text_dict.load(dict_file)

    model = BiLSTMModel(text_dict.id_size, attention=attention)
    trainer = tftrainer.Trainer(model)
    trainer.train(
        name,
        workpath,
        epoch,
        train_params_func=lambda p: {"input_files": train_files},
        eval_params_func=lambda p: {"input_files": eval_files},
        )

def predict(name, attention, workpath, dict_file, input_files, with_score):
    """Predict by the model
    """
    input_files = tftrainer.path.findfiles(input_files)
    if not input_files:
        raise ValueError("Require input files")

    print("Input files:", file=sys.stderr)
    for filename in input_files:
        print("\t%s" % filename, file=sys.stderr)

    text_dict = TextDictionary()
    text_dict.load(dict_file)

    model = BiLSTMModel(text_dict.id_size, attention=attention)
    trainer = tftrainer.Trainer(model)
    for result in trainer.predict(name, workpath, params_func=lambda p: {"input_files": input_files}):
        line_nos, scores = result.outputs["line_no"], result.outputs["score"]
        for line_no, score in zip(line_nos, scores):
            if with_score:
                print("%s\t%d\t%.4f" % (line_no, 1 if score >= 0.5 else 0, score))
            else:
                print("%s\t%d" % (line_no, 1 if score >= 0.5 else 0))

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
        preproc_parser.add_argument("-d", "--dict-file", dest="dict_file", default="dict.data", help="The dictionary filename")
        preproc_parser.add_argument("--jieba", dest="use_jieba", action="store_true", default=False, help="Use jieba")
        preproc_parser.add_argument("--no-fit", dest="no_fit", action="store_true", default=False, help="Do not fit dictionary")

        train_parser = sub_parsers.add_parser("train", help="Train")
        train_parser.add_argument("-n", "--name", dest="name", required=True, help="Model name")
        train_parser.add_argument("--attention", dest="attention", action="store_true", default=False, help="Use attention")
        train_parser.add_argument("-d", "--dict-file", dest="dict_file", default="dict.data", help="The dictionary filename")
        train_parser.add_argument("--train-file", dest="train_files", required=True, help="The train input file(s)")
        train_parser.add_argument("--eval-file", dest="eval_files", help="The evaluate input file(s)")
        train_parser.add_argument("-p", "--workpath", dest="workpath", default="outputs", help="The workpath")
        train_parser.add_argument("-e", "--epoch", dest="epoch", type=int, default=100, help="The epoch number")

        predict_parser = sub_parsers.add_parser("predict", help="Predict")
        predict_parser.add_argument("-n", "--name", dest="name", required=True, help="Model name")
        predict_parser.add_argument("--attention", dest="attention", action="store_true", default=False, help="Use attention")
        predict_parser.add_argument("-d", "--dict-file", dest="dict_file", default="dict.data", help="The dictionary filename")
        predict_parser.add_argument("-f", "--file", dest="files", required=True, help="The input file(s)")
        predict_parser.add_argument("-p", "--workpath", dest="workpath", default="outputs", help="The workpath")
        predict_parser.add_argument("--with-score", dest="with_score", action="store_true", help="Print result with score")

        return parser.parse_args()

    def main():
        """The main entry
        """
        args = get_args()

        if args.action == "preproc":
            preproc(args.input, args.output, args.dict_file, args.use_jieba, args.no_fit)
        elif args.action == "train":
            train(args.name, args.attention, args.workpath, args.dict_file, args.train_files, args.eval_files, args.epoch)
        elif args.action == "predict":
            predict(args.name, args.attention, args.workpath, args.dict_file, args.files, args.with_score)
        else:
            raise ValueError("Unknown action [%s]" % args.action)

    main()
