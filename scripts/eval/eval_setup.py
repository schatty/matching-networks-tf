import numpy as np
import tensorflow as tf
tf.config.gpu.set_per_process_memory_growth(True)
from tqdm import tqdm
from matchnet.data import load
from matchnet.models import MatchingNetwork


def eval(config):
    np.random.seed(2019)
    tf.random.set_seed(2019)
    
    # Determine device
    if config['data.cuda']:
        cuda_num = config['data.gpu']
        device_name = f'GPU:{cuda_num}'
    else:
        device_name = 'CPU:0'

    data_dir = config['data.dataset_path']
    ret = load(data_dir, config, ['test'])
    test_loader = ret['test']

    # Setup training operations
    way = config['data.test_way']
    lstm_dim = config['model.lstm_size']
    w, h, c = list(map(int, config['model.x_dim'].split(',')))

    # Metrics to gather
    test_loss = tf.metrics.Mean(name='test_loss')
    test_acc = tf.metrics.Mean(name='test_accuracy')

    model = MatchingNetwork(way, w, h, c, lstm_size=lstm_dim)
    model.load(config['model.save_dir'])

    def calc_loss(x_support, y_support, x_query, y_query):
        loss, acc = model(x_support, y_support, x_query, y_query)
        return loss, acc

    with tf.device(device_name):
        for i_episode in tqdm(range(config['data.episodes'])):
            x_support, y_support, x_query, y_query = test_loader.get_next_episode()
            loss, acc = calc_loss(x_support, y_support, x_query, y_query)
            test_loss(loss)
            test_acc(acc)

    print("Loss: ", test_loss.result().numpy())
    print("Accuracy: ", test_acc.result().numpy())
