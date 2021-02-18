import numpy as np
import matplotlib.pyplot as plt
import experiment
import interpret
import cosmology
import consensus
import pickle
import copy
import yaml
import ipdb


class Configuration():
    def __init__(self, config_file = None):
        self.config_dict = self.read(config_file)
        pass

    def read(self, config):
        """
        Reads config file into a dictionary and returns it.

        Args:
            config (str): Name of config file.

        Returns:
            config_dict (dict): Dictionary containing config information.
        """

        with open(config, 'r') as config_file_obj:
            config_dict = yaml.safe_load(config_file_obj)

        return config_dict


    def check_configuration(self,config):
        '''
        Checks that the configuration file makes sense, is compliant.
        '''
        necessary_params = ['']