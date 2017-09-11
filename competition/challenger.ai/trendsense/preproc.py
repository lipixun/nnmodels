#!/usr/bin/env python
# encoding=utf8
# pylint: disable=wrong-import-position

""" Preprocess the data
    Author: lipixun
    File Name: preproc.py
    Description:

        This script does the following things:

            1. Split data into train and validate dataset
            2. Train feature normalization model
            3. Run feature normalization
            4. Generate hdf5 file (for training / predicting)

"""

import sys
reload(sys)
sys.setdefaultencoding("utf8")

import logging

from cPickle import load, dump

import h5py
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, Imputer

TestERA = 10.0
BatchSize = 512

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO)
logger = logging.getLogger("preprocessor")

def runTrain(args):
    """Run preprocess for training
    """
    logger.info("Load input file: %s", args.input)
    df = pd.read_csv(args.input, dtype=np.float32, engine="c") # pylint: disable=no-member
    # Split data into train and test
    data = df.values
    train = data[data[:,-1] != TestERA]
    trainX = train[:,1:-4]
    trainW = train[:,-4]
    trainY = train[:,-3]
    test = data[data[:,-1] == TestERA]
    testX = test[:,1:-4]
    testW = test[:,-4]
    testY = test[:,-3]
    logger.info("Get Train data X%s W%s Y%s Test data X%s W%s Y%s", trainX.shape, trainW.shape, trainY.shape, testX.shape, testW.shape, testY.shape)
    # Create the preprocessors
    pl = Pipeline([ ("scaler", RobustScaler()), ("imputer", Imputer()) ])
    _trainX = pl.fit_transform(trainX)
    _testX = pl.transform(testX)     # NOTE: MUST not fit by the test data
    # Show some examples
    print "="*10, "TrainX Before Transform", "="*10
    print trainX[0]
    print "="*10, "TrainX After Transform", "="*10
    print _trainX[0]
    # Write hdf5 file
    logger.info("Write hdf5 file: %s", args.output)
    h5f = h5py.File(args.output, "w")
    h5f.create_dataset("x", data=_trainX, chunks=(BatchSize, _trainX.shape[1]))
    h5f.create_dataset("w", data=trainW, chunks=(BatchSize, ))
    h5f.create_dataset("y", data=trainY, chunks=(BatchSize, ))
    h5f.create_dataset("test-x", data=_testX, chunks=(BatchSize, _testX.shape[1]))
    h5f.create_dataset("test-w", data=testW, chunks=(BatchSize, ))
    h5f.create_dataset("test-y", data=testY, chunks=(BatchSize, ))
    h5f.close()
    # Write model file
    logger.info("Write model file: %s", args.model)
    with open(args.model, "wb") as fd:
        dump(pl, fd)

def runApply(args):
    """Run preprocess for applying
    """
    # Load model
    logger.info("Load model file: %s", args.model)
    with open(args.model, "rb") as fd:
        pl = load(fd)
    # Load file
    logger.info("Load input file: %s", args.input)
    df = pd.read_csv(args.input, dtype=np.float32, engine="c") # pylint: disable=no-member
    X = pl.tranform(df.values[:,:-1])
    logger.info("Get data X%s", X.shape)
    # Write file
    logger.info("Write hdf5 file: %s", args.output)
    h5f = h5py.File(args.output, "w")
    h5f.create_dataset("x", data=X, chunks=(BatchSize, X.shape[1]))
    h5f.close()

if __name__ == "__main__":

    from argparse import ArgumentParser

    def getArguments():
        """Get the arguments
        """
        parser = ArgumentParser(description="preprocessor")
        subParsers = parser.add_subparsers(dest="action")
        # Train
        trainParser = subParsers.add_parser("train", help="Run train preprocess")
        trainParser.add_argument("-i", "--input", dest="input", required=True, help="The input csv file")
        trainParser.add_argument("-m", "--model", dest="model", default="preproc-model.bin", help="The preprocess model")
        trainParser.add_argument("-o", "--output", dest="output", required=True, help="The output hdf5 file")
        # Apply
        applyParser = subParsers.add_parser("apply", help="Apply preprocess logic to data")
        applyParser.add_argument("-i", "--input", dest="input", required=True, help="The input csv file")
        applyParser.add_argument("-m", "--model", dest="model", default="preproc-model.bin", help="The preprocess model")
        applyParser.add_argument("-o", "--output", dest="output", required=True, help="The output hdf5 file")
        # Done
        return parser.parse_args()

    def main():
        """The main entry
        """
        args = getArguments()
        if args.action == "train":
            runTrain(args)
        else:
            runApply(args)

    main()
