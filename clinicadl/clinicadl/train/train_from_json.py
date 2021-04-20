"""
Retrain a model defined by a commandline.json file
"""

import argparse

from ..tools.deep_learning import read_json, check_and_complete
from .train_autoencoder import train_autoencoder
from .train_singleCNN import train_single_cnn
from .train_multiCNN import train_multi_cnn


def retrain(json_path, output_dir):
    options = argparse.Namespace()
    options = read_json(options, json_path=json_path, read_computational=True)
    check_and_complete(options)
    options.output_dir = output_dir

    if options.network_type == "autoencoder":
        train_autoencoder(options)
    elif options.network_type == "cnn":
        train_single_cnn(options)
    elif options.network_type == "multicnn":
        train_multi_cnn(options)
