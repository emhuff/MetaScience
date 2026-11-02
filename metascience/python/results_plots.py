import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import experiment
import interpret
import cosmology
import consensus
import dill
import copy
import ipdb
import glob2

names = glob2.glob('*.dill')
#names=['MostlyBetOnThemConsensus-results.dill','ImpatientConsensus-results.dill', 'MostlyBetOnMeConsensus-results.dill']
##filename = sys.argv[1]

# 1. Get results file name from argv[1]
# - or get yaml from argv[1] and filename from yaml
# - or make this a function called at end of metascience_simulation script
# 2. Read dill results file
# To read:
'''
with open('data.dill', 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    data = dill.load(f)
'''
# 3. Plot the things

# Parameter history plot
# - in the theme of the Hubble Constant measurements over time
# - use the (to be created) results.xxx_history objects
plt.clf()
colors=[['#87255B', '#B370B0', '#BBA0CA', '#AA89E2'],['#BA5624', '#F06E17', '#F06E17', '#FFA552']]

names.reverse
labs=names
for fc, filename in enumerate(names):
    print(fc, filename, 'hihh')
    with open(filename, 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
        data = dill.load(f)

    histories = data.histories
    truth=data.true_parameters
    print(truth)
    print('----Plotting begins-----')
    # There is one history per interpreter

    # for count, history in enumerate(histories):
    #     print(f"Count {count} in {filename} \n")
    #     #print('Parameter history: ', history['cosmological_parameter_history'])

    #     #print('Systematics parameter history: ', history['systematics_parameter_history'])
    
    #print('Model history: ', histories[0]['cosmological_model_history'])


    for j,hist in enumerate(histories): # looping over interpreters
        
        fignum=j
        
        for valc, val in enumerate(hist['cosmological_parameter_history']):
            if histories[j]['cosmological_model_history'][valc]=='Cosine cosmology':
                marker='*'
            else:
                marker='d'
            
            #print(f"Interpreter {j} and step {valc} for method {filename}")
            if valc==0:
                plt.figure(fignum)
                print(f"iteration {valc},interpretor {j} method {filename} fignum {fignum} huh")
                plt.plot(valc,val, marker=marker, markersize=10.0,color=colors[j][fc],alpha=0.9, label='Interpreter %i for approach %s'%(j,filename[:-13]))
                #plt.plot(valc,histories[j]['systematics_parameter_history'][valc][0], markersize=10.0,alpha=0.9,marker=marker, linestyle='-',color=colors[j][fc+2], label='Interpreter %i for approach %s'%(j,names[j]))
            else: 
                plt.figure(fignum)  
                plt.plot(valc,val, marker=marker, color=colors[j][fc],markersize=10.0,alpha=0.9)
                #plt.plot(valc,histories[j]['systematics_parameter_history'][valc][0],markersize=10.0,alpha=0.9, marker=marker,  linestyle='-',color=colors[j][fc+2])

    #    plt.plot(histories[0]['cosmological_parameter_history'], label='%s  - count 0'%filename[:-13])
    #    plt.plot(histories[1]['cosmological_parameter_history'], label='%s  - count 1'%filename[:-13])
        #plt.hlines(0.5, 0, len(histories[0]['cosmological_parameter_history']), color='k', linestyles='dashed')


    fig = plt.figure(fc)       
    plt.xlabel('Iteration number')
    plt.ylabel('Cosmological parameter value')
    plt.legend(loc='center')
    pl.savefig('cosmology_convergence_summary_interpretor%i.png'%fc)
