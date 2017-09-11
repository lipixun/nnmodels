#!/usr/bin/env python
# encoding=utf8
# pylint: disable=wrong-import-position

""" Generate the commit data
    Author: lipixun
    File Name: commit.py
    Description:

        This script is used to generate the data to commit

"""

import sys
reload(sys)
sys.setdefaultencoding("utf8")

import pandas as pd

def commit(args):
    """Generate the data to commit
    """
    df = pd.read_csv(args.input, engine="c") # pylint: disable=no-member
    results = []
    with open(args.result, "rb") as fd:
        for content in fd:
            results.append(float(content))
    with open(args.output+".csv", "wb") as fd:
        print >>fd, "id,proba"
        for id, value in zip(df.values[:,0], results):
            print >>fd, "%d,%f" % (int(id), value)

if __name__ == "__main__":

    from argparse import ArgumentParser

    def getArguments():
        """Get arguments
        """
        parser = ArgumentParser(description="Commit data generator")
        parser.add_argument("-i", "--input", dest="input", required=True, help="The input file")
        parser.add_argument("-r", "--result", dest="result", required=True, help="The result file")
        parser.add_argument("-o", "--output", dest="output", required=True, help="The output file")
        return parser.parse_args()

    def main():
        """The main entry
        """
        args = getArguments()
        return commit(args)

    main()
