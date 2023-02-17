import numpy as np
import yaml
from loguru import logger
import time
from datetime import datetime
import os
import shutil
import pickle
from copy import deepcopy

class Monitor:
    def __init__(self):
        self.data = dict()
        self.name_set = set()

    def update(self, **kwargs):
        for name, value in kwargs.items():
            if name not in self.name_set:
                self.data[name] = []
                self.name_set.add(name)
            self.data[name].append(deepcopy(value))
            
            
def add_log(log_root_path):
    now = datetime.now()
    date_time = now.strftime('%Y-%m-%d__%H-%M-%S')
    log_path =log_root_path + date_time
    os.mkdir(log_path) 
    logger.add(log_path + '/log.log')
    return log_path