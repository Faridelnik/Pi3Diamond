###############
# stimulation
################
from chaco import shell

##########  Rabi ramsy rabi pi messung ##################
############################################################

# tau = np.arange(0., 100., 1.)

# sequence = [(['sequence'], 100)]

# for t in tau:
   

   # # #rabi pi-messung
# # #tau = np.arange(0., 750., 75.)
# # #
# # #sequence = [(['sequence'], 100)]
# # #
# # #for t in tau:
# # #   x=int(t/75)
# # #   for i in range(x):
# # #       sequence +=  [ (['microwave'],75),([],20)]
# # #   sequence +=[(['laser', 'detect'],3000), ([],1000) ]

    # # # ramsay
# # #   sequence += [ (['microwave_b'],45),([],t), (['microwave_b'],45),(['laser', 'detect'],3000), ([],1000) ]
    
    # # # rabi
    # sequence += [ (['microwave'],t), (['laser', 'detect'],3000), ([],1000) ]

# # #sequence = 100*[ (['microwave'],100), ([],10000) ]

#######Echo############
# tau=np.arange(15500.,21500.,200.)
# sequence=[(['sequence'], 100)]
# pipuls_a= 76.
# for t in tau:
    # sequence+=[(['microwave'],pipuls_a/2),([],18500), (['microwave'],pipuls_a),([],t), (['microwave'],pipuls_a/2), (['laser', 'detect'],3000), ([],1000) ]
    # sequence+=[(['microwave'],pipuls_a/2),([],18500), (['microwave'],pipuls_a),([],t), (['microwave'],pipuls_a*3./2), (['laser', 'detect'],3000), ([],1000) ]
################## deutsch-jozsa##################
#############################################



# ##########  echo #################
tau = np.arange(0,1000,1.)
pipuls_a= 76.
pipuls_b=72.
sequence = [(['sequence'], 100)]
for t in tau:
    # # f1
    sequence += [ (['microwave'],pipuls_a/2), ([],324), ([],t),(['microwave'],pipuls_a/2), (['laser', 'detect'],3000), ([],1000) ]

    # # # f2
    sequence += [ (['microwave'],pipuls_a*5/2.), ([],(324-2*pipuls_a)),([],t), (['microwave'],pipuls_a/2), (['laser', 'detect'],3000), ([],1000) ]

    # # # f3
    sequence += [ (['microwave'],pipuls_a/2), ([],(296-pipuls_b*2)), (['microwave_b'],pipuls_b*2), ([],28), ([],t),(['microwave'],pipuls_a/2), (['laser', 'detect'],3000), ([],1000) ]

    # # # f4
    sequence += [ (['microwave'],pipuls_a*5/2.), (['microwave_b'],pipuls_b*2),([],28), ([],t), (['microwave'],pipuls_a/2), (['laser', 'detect'],3000), ([],1000) ]

    #sequence += [ ([],t),(['laser', 'detect'],3000), ([],1000) ]

    #sequence += [ (['microwave'],pipuls_a), ([],t),(['microwave'],pipuls_a*3./2),(['laser', 'detect'],3000), ([],1000) ]

    # sequence += [(['microwave'],pipuls_a/2), (['laser', 'detect'],3000), ([],1000) ]

    # sequence += [(['microwave_b'],pipuls_b), (['laser', 'detect'],3000), ([],1000) ]

    # sequence += [(['microwave_b'],pipuls_b/2), (['laser', 'detect'],3000), ([],1000) ]
    
    
    


    ########## mit echo powera= 5dbm powerb=3.75 #################
# tau = np.arange(0., 5000., 10.)

# sequence = [(['sequence'], 100)]
    
# for t in tau:
    # # f1
    # # sequence += [ (['microwave'],51.25), ([],410), ([],30), (['microwave'],102.5), ([],t), (['microwave'],51.25), (['laser', 'detect'],3000), ([],1000) ]

    # # f2
    # #sequence += [ (['microwave'],256.25), ([],205), ([],30),(['microwave'],102.5), ([],t), (['microwave'],51.25), (['laser', 'detect'],3000), ([],1000) ]

    # # f3
     # sequence += [ (['microwave'],51.25), ([],205), (['microwave_b'],205), ([],30),(['microwave'],102.5), ([],t), (['microwave'],51.25), (['laser', 'detect'],3000), ([],1000) ]

    # # f4
    # #sequence += [ (['microwave'],256.25), (['microwave_b'],205),([],30), (['microwave'],102.5), ([],t), (['microwave'],51.25), (['laser', 'detect'],3000), ([],1000) ]

    
    
    # ########## mit echo powera= 6dbm powerb=6 #################
# tau = np.arange(0., 1000., 10.)

# sequence = [(['sequence'], 100)]
# pipuls=90. 
    
# for t in tau:
    # # # f1
    # #sequence += [ (['microwave'],pipuls/2), ([],pipuls*4), (['microwave'],pipuls), ([],t), (['microwave'],pipuls/2), (['laser', 'detect'],3000), ([],1000) ]

    # # # f2
    # sequence += [ (['microwave'],pipuls*5/2.), ([],pipuls*2),(['microwave'],pipuls), ([],t), (['microwave'],pipuls/2), (['laser', 'detect'],3000), ([],1000) ]

    # # # f3
    # #sequence += [ (['microwave'],pipuls/2), ([],pipuls*2), (['microwave_b'],pipuls*2),(['microwave'],pipuls), ([],t), (['microwave'],pipuls/2), (['laser', 'detect'],3000), ([],1000) ]

    # # # f4
    # # sequence += [ (['microwave'],pipuls*5/2), (['microwave_b'],pipuls*2), (['microwave'],pipuls), ([],t), (['microwave'],pipuls/2), (['laser', 'detect'],3000), ([],1000) ]

    
    
power_a = 9 # dBm
power_b = 9 # dBm

frequency_a = 2.73170e9 # Hz
frequency_b = 3.0096e9 # Hz

microwave.setOutput(power_a, frequency_a)
microwave_2.setOutput(power_b, frequency_b)

pulse_generator.Sequence( sequence )

###############
# acquisition
################

n_bins = 3000

binwidth = 1000 # ps

n_detect =4* len(tau) # number of laser pulses

pulsed = time_tagger.Pulsed(n_bins, binwidth, n_detect, 1, 2, 3)

from analysis.pulsed import spin_state

def fetch():
    mat = pulsed.getData()
    y, profile, edge = spin_state(mat, 1.0, 300.)
    return y
    
def plot():
    shell.close('all')
    x=fetch().reshape((len(tau),4))
    shell.plot(tau,x[:,0],'-b')
    shell.hold(True)
    shell.plot(tau,x[:,1],'-r')
    shell.plot(tau,x[:,2],'-g')
    shell.plot(tau,x[:,3],'-c')
    

    