import yaml

from lib.parser import parse_arguments
from train import train


def main():
    args = parse_arguments()

    with open(args.config, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    
    print(f'Starting: {config["task"]} \nModel-type: {config["model_type"]}')

    if config['task'] == 'train': train(config, debug=args.debug)
    elif config['task'] == 'test': print('to be implemented...')


if __name__ == '__main__':
    main()