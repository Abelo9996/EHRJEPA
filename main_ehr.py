#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Main entry point for JEPA-EHR training

import argparse
import logging
import pprint
import yaml

from train_ehr import main as app_main

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser(description='JEPA-EHR Training')
parser.add_argument(
    '--fname', type=str,
    help='name of config file to load',
    default='configs/mimic_ehr_base.yaml')


if __name__ == '__main__':
    args = parser.parse_args()

    logger.info(f'Loading config from {args.fname}')

    # -- load script params
    params = None
    with open(args.fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        logger.info('Loaded params:')
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)

    logger.info('Starting JEPA-EHR training...')
    app_main(args=params)
