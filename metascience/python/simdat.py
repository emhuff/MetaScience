import astropy
import numpy as np
import matplotlib.pyplot as plt


def gen_distance(model={'Name':'FlatLambdaCDM', 'params': ['H0','Om0'], 'values':[70,0.3]}, redshift=np.arange(0.1,1,0.1)):
#    name = model['Name']
#    print(name)
#    import astropy
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(70, 0.3)
    mu = cosmo.distmod(redshift).value
    return mu

def gen_error(model={'Distribution':'Gaussian', 'params':['mean', 'sigma'], 'values':[0,5]},redshift=np.arange(0.1,1,0.1)):

    mu = gen_distance(redshift=redshift)

    mu_err = model['values'][0] + model['values'][1]*np.random.randn(len(redshift))
    
    return mu, mu_err
    


def gen_systematic(redshift=np.arange(0.1,1,0.1)):

    mu = gen_distance(redshift=redshift)

    bias = mu*(1+redshift**2)

    return bias


z = redshift=np.arange(0.1,1,0.1)
mu, mu_err = gen_error(redshift=z)

bias = gen_systematic(redshift=z)

plt.errorbar(z,mu,mu_err, marker='.', linestyle='None', label='original')

plt.errorbar(z,mu+bias,mu_err, marker='.', color='r', linestyle='None', label='biased')

plt.xlabel('z')
plt.ylabel('mu')
plt.savefig('test.png')


