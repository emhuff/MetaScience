from abc import ABCMeta, abstractmethod

class Experiment():
    def __init__(self):
        pass

    @property
    def cosmology():
        '''
        Need to specify the cosmology.
        '''
        raise NotImplementedError()

    @property
    def systematics():
        '''
        Need to specify the systematic error model.
        '''
        raise NotImplementedError()

    @property
    def noise():
        '''
        Need to specify the noise model.
        '''
        raise NotImplementedError()

    @property
    def kind():
        '''
        What experiment is this? Know thyself, object.
        '''

    @abstractmethod
    def _get_ideal_data_vector():
        # generate the ideal data vector from the cosmology.
        pass

    @abstractmethod
    def _add_systematics():
        pass

    @abstractmethod
    def _add_noise():
        pass

    def generate_data():
        self._make_ideal_data_vector():
        self._add_noise()
        self._add_systematics()


class CMBExperiment(Experiment):
    def __init__(self,cosmology = None, systematics_model = None, noise_model = None):
        super().__init__()
        self.kind = 'CMB'
        self.cosmology = cosmology

    def _set_systematics(systematics_model):
        pass

    def _set_noise(noise_model):
        pass


class PendulumExperiment(Experiment):
    def __init__(self,cosmology = None, systematics_model = None, noise_model = None):
        super().__init__()
        self.kind = 'pendulum'
        self.cosmology = cosmology

    def _set_systematics(systematics_model):
        pass

    def _set_noise(noise_model):
        pass
