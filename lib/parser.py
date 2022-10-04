import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-config', type=str)

    return parser.parse_args()