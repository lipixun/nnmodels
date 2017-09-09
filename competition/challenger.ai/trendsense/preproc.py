#!/usr/bin/env python
# encoding=utf8

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

if __name__ == "__main__":

    from argparse import ArgumentParser

    def getArguments():
        """Get the arguments
        """
        parser = ArgumentParser(description="preprocessor")
        subParsers = parser.add_subparsers(dest="action")
        # Train
        trainParser = subParsers.add_parser("train", help="Run train preprocess")
        # Apply
        applyParser = subParsers.add_parser("apply", help="Apply preprocess logic to data")
        # Done
        return parser.parse_args()

    def main():
        """The main entry
        """
        args = getArguments()

    main()
