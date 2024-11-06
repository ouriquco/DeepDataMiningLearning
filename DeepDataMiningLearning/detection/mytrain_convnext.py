import datetime
import os
import time
import math
import sys
import torch
import torch.utils.data
import torchvision
import torchvision.models.detection
import argparse
import utils

from trainutils import create_aspect_ratio_groups, GroupedBatchSampler
from dataset import get_dataset
from models import create_detectionmodel
from myevaluator import simplemodelevaluate, modelevaluate
from torchinfo import summary

MACHINENAME='HPC'
USE_AMP=True #AUTOMATIC MIXED PRECISION

def arg_parse():
    pass

def main(args):
    pass