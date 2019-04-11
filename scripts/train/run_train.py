import argparse
import configparser

from train_setup import train


def preprocess_config(c):
    conf_dict = {}
    int_params = ['data.train_way', 'data.test_way', 'data.train_support',
                  'data.test_support', 'data.train_query', 'data.test_query',
                  'data.query', 'data.support', 'data.way', 'data.episodes',
                  'data.gpu', 'data.cuda', 'model.lstm_size', 'train.epochs',
                  'train.patience', 'data.batch']
    float_params = ['train.lr']
    for param in c:
        if param in int_params:
            conf_dict[param] = int(c[param])
        elif param in float_params:
            conf_dict[param] = float(c[param])
        else:
            conf_dict[param] = c[param]
    return conf_dict


parser = argparse.ArgumentParser(description='Run training')
parser.add_argument("--config", type=str, default="./scripts/omniglot.conf",
                    help="Path to the config file.")

# Run training
args = vars(parser.parse_args())
config = configparser.ConfigParser()
config.read(args['config'])
config = preprocess_config(config['TRAIN'])
train(config)
