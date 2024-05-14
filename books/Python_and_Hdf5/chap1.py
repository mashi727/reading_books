

#%%
import numpy as np
temperature = np.random.random(1024)
dt = 10.0
start_time = 1375204299 # in Unix time
station = 15
np.savez("weather.npz", data=temperature, start_time=start_time, station=
station)
#%%

'''
So far so good. But what if we have more than one quantity per station? Say there’s also
wind speed data to record?
'''

#%%
out = np.load("weather.npz")
print(out["data"])
print(out["start_time"])
print(out["station"])

#%%

#%%

import h5py
with h5py.File("weather.hdf5", "w") as f:
    grp = f.create_group("{:010}".format(time))
    f["/15/temperature"] = temperature
    f["/15/temperature"].attrs["dt"] = 10.0
    f["/15/temperature"].attrs["start_time"] = 1375204299
    #f["/15/wind"] = wind
    f["/15/wind"].attrs["dt"] = 5.0
    #f["/20/temperature"] = temperature_from_station_20
#%%

#%%
with h5py.File("test.hdf5", "w") as f:
    grp = f.create_group('mygroup')
    dset = grp.create_dataset('dataset', (100,))
#%%


