# Python helloworld
import argparse
# from trainer import test_final_trajectory as trainer_ae
from trainer import idea_test_demo2 as trainer_ae

import numpy as np
import random
import torch


def prepare_seed(rand_seed):
    np.random.seed(rand_seed)
    random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed_all(rand_seed)


def parse_config():
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument("--cuda", default=True)

    parser.add_argument("--past_len", type=int, default=8, help="length of past (in timesteps)")
    parser.add_argument("--future_len", type=int, default=12, help="length of future (in timesteps)")
    parser.add_argument("--dim_embedding_key", type=int, default=24)
    parser.add_argument("--data_scale", type=float, default=1)
    parser.add_argument("--data_scale_old", type=float, default=1.86)
    parser.add_argument("--train_b_size", type=int, default=512)
    parser.add_argument("--test_b_size", type=int, default=4096)
    parser.add_argument("--time_thresh", type=int, default=0)
    parser.add_argument("--dist_thresh", type=int, default=100)
    parser.add_argument("--mode", type=str, default='intention',
                        choices=['intention1', 'intention2', 'intention3', 'addressor_warm', 'addressor',
                                 'addressor_warm2', 'addressor2', 'addressor_warm3', 'addressor3', 'trajectory'],
                        help='Stage of training.')

    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument("--model_ae1", default='./training/training_ae/...')
    parser.add_argument("--model_ae2", default='./training/training_ae/...')
    parser.add_argument("--model_ae3", default='./training/training_ae/...')

    parser.add_argument("--reproduce", default=True)

    parser.add_argument("--dataset_file", default="SDD", help="dataset file")
    parser.add_argument("--info", type=str, default='', help='Name of training. '
                                                             'It will be used in tensorboard log and test folder')
    return parser.parse_args()


def main(config):
    if config.reproduce:
        prepare_seed(0)
        #  config.model_ae1 = "/home/featurize/work/training/model_encdec_trajectory.zip"
#         autodl-tmp/multi/training/training_addressor/2023-04-21_try1/model_ae_2023-04-21
        config.model_ae1 = './training/model_ae_8'
        config.model_ae2 = './training/model_ae_6'
        config.model_ae3 = './training/model_ae_4'
        print(config.model_ae1)
        print(config.model_ae2)
        print(config.model_ae3)
        t = trainer_ae.Trainer(config)
        t.fit()


if __name__ == "__main__":
    config = parse_config()
    main(config)
