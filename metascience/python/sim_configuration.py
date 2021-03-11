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
import pprint



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
        necessary_params = ['consensus']#, 'experiment', 'interpreter']
        pass


if __name__ == '__main__':

    test = Configuration(config_file='example.yaml')
    #pprint.pprint(test.config_dict['consensus'])

    consensus_sim = test.config_dict['consensus']
    # first consensus
#    print(consensus['consensusize'][0])
    #consensus models to testing
    consensus_names = consensus_sim.keys()
    for name in consensus_names:
        print(f'Consensus: {name}')
        kwargs = consensus_sim[name]
        interpreters='SimplePendulumExperimentInterpreter'
        this_consensus = getattr(consensus,name)(interpretations = interpreters, **kwargs)
        #this_consensus = getattr(consensus,model['name'        '])(interpretations = interpreters, **kwargs)
        #this_consensus.tension_metric()
        # now run stuff
        print(kwargs)
