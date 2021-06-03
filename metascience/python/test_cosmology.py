import cosmology
import numpy as np
import matplotlib.pyplot as plt

# Test the true cosmology.
truth = cosmology.LCDM_distanceModulus()

truth_parameters = truth.get_parameter_set()

truth_parameters[0]=70.
truth_parameters[1] = 0.26
truth_parameters[2] = 0.7
truth_parameters[3]  = 0.04
#truth_parameters[0] = 1.0 # oscillator frequency
#truth_parameters[1] = 0.01 # C -- drag/damping coefficient
#truth_parameters[2] = .0 # A -- amplitude of forcing function
#truth_parameters[3] = .1 # wd -- freq. of forcing function
#truth_parameters[4] = 0.0 # phid -- phase of forcing function
#truth_parameters[5] = 1.0  # theta_x0 -- initial position of pendulum
#truth_parameters[6] = 0.0 # theta_v0-- initial velocity of pendulum

# At what times are we generating the data?
times = np.linspace(0,1,5000)

# Now generate some data with this parameter set!
real_data = truth.generate_model_data_vector(times,parameters=truth_parameters)


# Test the model cosmology.
model =  cosmology.LCDM_distanceModulus()
()
model_parameters = model.get_parameter_set()
model_data = model.generate_model_data_vector(times,parameters=model_parameters)

# plot this.
plt.plot(times,real_data,label='real')
plt.plot(times,model_data,label='model')
plt.xlabel('time')
plt.ylabel('position')
plt.legend(loc='best')
plt.show()
