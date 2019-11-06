from __future__ import division
from __future__ import print_function
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str,
                    default='',
                    help='')
args = parser.parse_args()
name = args.name
with open(name,'r') as f:
    all_lines = f.readlines()
    print('total lines:', len(all_lines))
