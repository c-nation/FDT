"""Author: Charlie Nation

Loads and saves hdf5 files for FDT plots.

This is quite general, so you can pass in a data dictionary, and it saves it in an hdf5 file in the specified location.
You need this location, and the dictionary keys to load the file again.

These work by specifying a path inside the file. See e.g. http://docs.h5py.org/en/stable/quick.html"""

import h5py


class LoadSave:

    def __init__(self, filename, group):
        self.filename = filename  # name of hdf5 file ... include the path!
        self.group = group  # path in which data is to be stored. e.g 'parameters/init_state'

    def save_hdf5(self, data):
        """data is in dictionary: e.g. data = dict(times=times, obs_t=obs_t, gamma=gamma, ...) """

        with h5py.File(self.filename) as hdf:
            # check if this path exists
            e = self.group in hdf
            if not e:
                # If it doesn't exist create group
                g1 = hdf.create_group(self.group)
                for key in data.keys():
                    g1.create_dataset(key, data=data[key])

                print('saved dataset in new group:')
                print(self.group)
                print('file: ', self.filename)

            else:
                # Otherwise just create the datasets
                g1 = hdf.get(self.group)
                for key in data.keys():
                    g1.create_dataset(key, data=data[key])
                print('Saved dataset in existing group:')
                print(self.group)
                print('file: ', self.filename)

    def load_hdf5(self, data_keys):
        """data_names must be a list of the names of dictionary elements of data.

                e.g.
                data_names = ['times', 'obs_t', 'gamma',...]
                for data = dict(times=times, obs_t=obs_t, gamma=gamma, ...) """

        with h5py.File(self.filename) as hdf:
            # check if this path exists
            e = self.group in hdf
            if not e:
                # If it doesn't exist
                raise Exception('DATA DOES NOT EXIST:', 'Group: ', self.group, ', file: ', self.filename)

            else:
                g1 = hdf.get(self.group)
                data = dict()
                for key in data_keys:
                    print(key)
                    data[key] = g1.get(key)[()]

        return data

    def check_hdf5(self):
        """Checks if particular dataset exists in hdf5 file - returns true if file exists"""
        import os
        if os.path.isfile(self.filename):
            with h5py.File(self.filename) as hdf:
                # check if this path exists
                e = self.group in hdf
                print('Checking: ', self.filename, '-', self.group)
                if not e:
                    checker = False
                    print('Data Does Not Exist: Checker = False')
                else:
                    print('Data Exists: Checker = True')
                    checker = True
        else:
            checker = False
            print('Data Does Not Exist: Checker = False')
        return checker

    def list_groups_hdf5(self):
        """Lists all groups in filename"""

        with h5py.File(self.filename) as hdf:
            print(list(hdf.keys()))

    def list_everything_hdf5(self, include_dataset_info=False):
        """Lists all groups, subgroups and datasets in filename...
        This can be a big output..."""

        def h5py_dataset_iterator(g, prefix=''):
            for key in g.keys():
                item = g[key]
                path = '{}/{}'.format(prefix, key)
                if isinstance(item, h5py.Dataset):  # test for dataset
                    yield (path, item)
                elif isinstance(item, h5py.Group):  # test for group (go down)
                    yield from h5py_dataset_iterator(item, path)

            with h5py.File(self.filename) as f:
                for (path, dset) in h5py_dataset_iterator(f):
                    if include_dataset_info:
                        print(path, dset)
                    else:
                        print(path)
