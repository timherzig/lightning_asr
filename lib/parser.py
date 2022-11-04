import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-config', type=str)
    parser.add_argument('-debug', type=bool, default=False)

    return parser.parse_args()