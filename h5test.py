import h5py
import numpy as np

array = np.ones((122,122,3))

print(array)
with h5py.File("mytestfile.hdf5", "w") as f:
	f.create_dataset("mydataset", data = array, chunks = True, compression = 'gzip')


with h5py.File("mytestfile.hdf5", "r") as f:
	print(f['mydataset'][:])
