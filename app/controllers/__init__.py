import os
import sys
import yaml

"""

Setting the path to the config file

"""
CONFIG_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'config', 'llm.yaml')
def set_config_location():
    with open(CONFIG_FILE_PATH, 'r') as file:
        config = yaml.safe_load(file)
    return config