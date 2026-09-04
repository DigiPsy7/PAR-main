import argparse
import os
import logging
import torch
import time


class logging_Config():
    def __init__(self, name):
        # creating a logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s | %(levelname)s -> %(message)s')

        file_handler = logging.FileHandler(f'{name}.log')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)

        self.logger.addHandler(file_handler)
        # logger.info('this is a log message...')

    def get_config(self):
        return self.logger


if __name__ == '__main__':
    data_parser = argparse.ArgumentParser('arguments for data')
    data_parser.add_argument('--dataset_name', default='mnist', help='["mnist", "cifar", "fashion-mnist"]')
    data_parser.add_argument('--cls_to_erased', default=[0, 1, 2, 3]) #[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
    data_parser.add_argument('--batch_size', default=128)


    training_parser = argparse.ArgumentParser('arguments for training')
    training_parser.add_argument('--communication_rounds', default=50)
    training_parser.add_argument('--repair_rounds', default=10)
    training_parser.add_argument('--train_epochs', default=2)

    model_parser = argparse.ArgumentParser('arguments for model')
    model_parser.add_argument('--model_name', default='resnet', help='[resnet, allcnn, mobilenet]')
    model_parser.add_argument('--num_clients', default=5)
    model_parser.add_argument('--device', default='cuda')
    model_parser.add_argument('--federated_mode', default='fedavg', type=str, help='[no, part, all, per]')

    type_parser = argparse.ArgumentParser('arguments for type of training and evaluation')
    type_parser.add_argument('--retrain', default=False, type=bool)

    data_args = data_parser.parse_args()
    training_args = training_parser.parse_args()
    model_args = model_parser.parse_args()
    type_args = type_parser.parse_args()

    log_file = f'k_4_mnist-{model_args.num_clients}-{model_args.federated_mode}-{type_args.retrain}-{training_args.communication_rounds}-{training_args.repair_rounds}-{training_args.train_epochs}-{model_args.model_name}'
    logger = logging_Config(log_file).get_config()

    args = (model_args, data_args, training_args, type_args)

    assert data_args.dataset_name in ['mnist', 'cifar', 'fashion-mnist']
    from get_trainer import get_FL_trainer

    FL_trainer = get_FL_trainer(args, logger)
    
    # Single sample output
    # FL_trainer.get_sample_prob()

    FL_trainer.train()
    FL_trainer.load_ckpt(path='server.pt')
    # start_time = time.time()
    FL_trainer.impair_train()
    # end_time = time.time()
    # print('Time cost:', end_time - start_time)