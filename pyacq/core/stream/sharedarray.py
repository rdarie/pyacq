# -*- coding: utf-8 -*-
# Copyright (c) 2016, French National Center for Scientific Research (CNRS)
# Distributed under the (new) BSD License. See LICENSE for more info.

import numpy as np
import sys, random, string, tempfile, mmap, os
from multiprocessing import shared_memory


# TODO
# On POSIX system it can optionally the shm_open way to avoid mmap.
verbose = False

class SharedMem:
    """Class to create a shared memory buffer.
    
    This class uses mmap so that unrelated processes (not forked) can share it.
    
    It is usually not necessary to instantiate this class directly; use
    :func:`OutputStream.configure(transfermode='sharedmem') <OutputStream.configure>`.
    
    Parameters
    ----------
    size : int
        Buffer size in bytes.
    shm_id : str or None
        The id of an existing SharedMem to open. If None, then a new shared
        memory file is created.
    
    """
    def __init__(self, nbytes, shm_id=None):
        self.nbytes = nbytes
        self.shm_size = (self.nbytes // mmap.PAGESIZE + 1) * mmap.PAGESIZE
        self.shm_id = shm_id
        self.pid = os.getpid()
        if verbose:
            print('class SharedMem: nbytes = {}; id = {}'.format(self.nbytes, self))
        if shm_id is None:
            self.shm_id = u'pyacq_SharedMem_'+''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(32))
            if verbose:
                print('class SharedMem: writing; pid = {}; shm_size = {}\t\nshm_id = {}'.format(self.pid, self.shm_size, self.shm_id))
            self.shm = shared_memory.SharedMemory(name=self.shm_id, create=True, size=self.shm_size)
        else:
            if verbose:
                print('class SharedMem: reading; pid = {}; shm_size = {}\t\nshm_id = {}'.format(self.pid, self.shm_size, self.shm_id))
            self.shm = shared_memory.SharedMemory(name=self.shm_id, create=False)
    
    def close(self):
        """Close this buffer.
        """
        self.shm.close()
    
    def to_dict(self):
        """Return a dict that can be serialized and sent to other processes to
        access this buffer.
        """
        return {'nbytes': self.nbytes, 'shm_id': self.shm_id}
    
    def to_numpy(self, offset, dtype, shape, strides=None):
        """Return a numpy array pointing to part (or all) of this buffer.
        """
        return np.ndarray(
            buffer=self.shm.buf, shape=shape,
            strides=strides, offset=offset, dtype=dtype)        
        




class SharedArray:
    """Class to create shared memory that can be viewed as a `numpy.ndarray`.
    
    This class uses mmap so that unrelated processes (not forked) can share it.
    
    The parameters of the array may be serialized and passed to other processes
    using `to_dict()`::
    
        orig_array = SharedArray(shape, dtype)
        spec = pickle.dumps(orig_array.to_dict())
        shared_array = SharedArray(**pickle.loads(spec))
    
    
    Parameters
    ----------
    shape : tuple
        The shape of the array.
    dtype : str or list
        The dtype of the array (as understood by `numpy.dtype()`).
    shm_id : str or None
        The id of an existing SharedMem to open. If None, then a new shared
        memory file is created.
        On linux this is the filename, on Windows this is the tagname.
    
    """
    def __init__(self, shape=(1,), dtype='float64', shm_id=None):
        self.shape = shape
        self.dtype = np.dtype(dtype)
        nbytes = int(np.prod(shape) * self.dtype.itemsize)
        self.shmem = SharedMem(nbytes, shm_id)
    
    def to_dict(self):
        return {'shape': self.shape, 'dtype': self.dtype, 'shm_id': self.shmem.shm_id}
    
    def to_numpy(self):
        return np.frombuffer(self.shmem.shm.buf, dtype=self.dtype).reshape(self.shape)

