import pickle
M17data=open('C:/data/BP/tag4/rdj_ramsey.pys', 'wb')
dumpo={"notes":'f1, f2, f3 , f4, bright, dark mw1, pi/2mw1, dark mw2, pi/2mw2, x=fetch().reshape((len(tau),2), t1 fest 18500',"mw1": [frequency_a, power_a,pipuls_a],"mw2": [frequency_b, power_b],"data": pulsed.getData(), "counts": fetch(),"Zeit": tau}
#dumpo={"time": tau, "counts": fetch(), "mw1": [frequency_a, power_a]}
pickle.dump(dumpo, M17data)
M17data.close()