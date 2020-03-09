import os
import argparse
import logging

from py_apriori.apriori import Apriori, CollapsedCsvFileReader
from py_apriori.assoc import (calculate_confidence,
                              create_associations)

logging.basicConfig()
logger = logging.getLogger('apriori')
logger.setLevel(logging.WARN)


class Program:
    def __init__(self):
        self.data = []

    def parse_arguments(self):
        parser = argparse.ArgumentParser(description='A tutorial of argparse!')
        parser.add_argument('-i', '--input', dest='FILE', required=True,
                            help='input file with two matrices', metavar='FILE',
                            type=lambda x: self.is_valid_file(parser, x))
        parser.add_argument('-c', '--confidence', dest='confidence_level', required=False,
                            default=0.80,
                            help='confidence level for association generation')
        parser.add_argument('-s', '--support', dest='support_level', required=False,
                            default=0.20,
                            help='support level for support generation')
        parser.add_argument('-n', '--no-drop', dest='drop_below_support_level', required=False,
                            default=True, action='store_false',
                            help='DO NOT drop transactions below support level for support generation')

        self.args = parser.parse_args()

    def is_valid_file(self, parser, arg):
        if not os.path.exists(arg):
            parser.error("The file %s does not exist!" % arg)
        else:
            return os.path.abspath(arg)

    @property
    def FILE(self):
        return self.args.FILE


def main():
    prog = Program()
    prog.parse_arguments()

    file_reader = CollapsedCsvFileReader(prog.FILE)
    raw_transactions = file_reader.read()

    apriori_instance = Apriori(raw_transactions)

    # setup for output
    support = prog.args.support_level
    confidence = prog.args.confidence_level
    drop_trans = prog.args.drop_below_support_level

    print("for this run we are using the following\n")
    print("\tSupport: {}".format(support))
    print("\tConfidence: {}".format(confidence))
    print("\tDrop Trans: {}".format(drop_trans))

    # just generate the levels and filter as needed
    support_level_output = apriori_instance.generate_levels(support_level=support, drop_below_support=drop_trans)
    print(support_level_output)

    # create the associations
    # TODO: mabye encapsulate this step.
    pc = create_associations(support_level_output)

    # generate the confidence levels.
    f = calculate_confidence(pc, confidence_level=confidence)

    print(f)

    print(len(f))


if __name__ == "__main__":
    main()
