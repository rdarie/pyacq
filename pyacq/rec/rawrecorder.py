# -*- coding: utf-8 -*-
# Copyright (c) 2016, French National Center for Scientific Research (CNRS)
# Distributed under the (new) BSD License. See LICENSE for more info.

import numpy as np
import collections
import logging
import os
import json
import pdb

from ..core import Node, register_node_type, ThreadPollInput, InputStream
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.util.mutex import Mutex

from ..version import version as pyacq_version

# extend the json.JSONEncoder class
class CustomEncoder(json.JSONEncoder):
    # overload method default
    def default(self, obj):
        # Match all the types you want to handle in your converter
        if isinstance(obj, np.dtype):
            try:
                return json.JSONEncoder.default(self, obj)
            except:
                if len(obj.descr) == 1:
                    return obj.name
                else:
                    return obj.descr
        else:
            # Call the default method for other types
            return json.JSONEncoder.default(self, obj)


class RawRecorder(Node):
    """
    Simple recorder Node of multiple streams in raw data format.
    
    Implementation is simple, this launch one thread by stream.
    Each one pull data and write it directly into a file in binary format.
    
    Usage:
    list_of_stream_to_record = [...]
    rec = RawRecorder()
    rec.configure(streams=list_of_stream_to_record, autoconnect=True, dirname=path_of_record)
    
    """

    _input_specs = {}
    _output_specs = {}
    
    def __init__(self, **kargs):
        Node.__init__(self, **kargs)
    
    def _configure(self, streams=[], autoconnect=True, dirname=None):
        self.streams = streams
        self.dirname = dirname
        
        assert not os.path.exists(dirname), 'dirname already exists'
        
        if isinstance(streams, list):
            names = ['input{}'.format(i) for i in range(len(streams))]
        elif isinstance(streams, dict):
            names = list(streams.keys())
            streams = list(streams.values())
        
        #make inputs
        self.inputs = collections.OrderedDict()
        for i, stream in enumerate(streams):
            name = names[i]
            input = InputStream(spec={}, node=self, name=name)
            self.inputs[name] = input
            if autoconnect:
                input.connect(stream)
                
    def _initialize(self):
        os.mkdir(self.dirname)
        self.files = []
        self.threads = []
        
        self.mutex = Mutex()
        
        self._stream_properties = collections.OrderedDict()
        
        for name, input in self.inputs.items():
            filename = os.path.join(self.dirname, name+'.raw')
            fid = open(filename, mode='wb')
            self.files.append(fid)
            
            flush_every_packet = (input.params.get('streamtype', None) == 'events')
            thread = ThreadRec(name, input, fid, flush_every_packet=flush_every_packet)
            self.threads.append(thread)
            thread.recv_start_index.connect(self.on_start_index)
            
            prop = {}
            for k in ('streamtype', 'dtype', 'shape', 'sample_rate', 'channel_info'):
                if k in input.params:
                    prop[k] = input.params[k]
            self._stream_properties[name] = prop
        
        self._stream_properties['pyacq_version'] = pyacq_version
        
        self._flush_stream_properties()
        
        self._annotations = {}
    
    def _start(self):
        for name, input in self.inputs.items():
            input.empty_queue()
        
        for thread in self.threads:
            thread.start()

    def _stop(self):
        for thread in self.threads:
            thread.stop()
            thread.wait()
        
        #test in any pending data in streams
        for i, (name, input) in enumerate(self.inputs.items()):
            ev = input.poll(timeout=0.2)
            if ev>0:
                pos, data = input.recv(return_data=True)
                self.files[i].write(data.tobytes())
        
        for f in self.files:
            f.close()

    def _close(self):
        pass
    
    def on_start_index(self, name, start_index):
        self._stream_properties[name]['start_index'] = start_index
        self._flush_stream_properties()
    
    def _flush_stream_properties(self):
        filename = os.path.join(self.dirname, 'stream_properties.json')
        with self.mutex:
            _flush_dict(filename, self._stream_properties)
    
    def add_annotations(self, **kargs):
        self._annotations.update(kargs)
        filename = os.path.join(self.dirname, 'annotations.json')
        with self.mutex:
            _flush_dict(filename, self._annotations)
    

def _flush_dict(filename, d):
    # pdb.set_trace()
    with open(filename, mode = 'w', encoding = 'utf8') as f:
        f.write(
            json.dumps(
                d, sort_keys=True, cls=CustomEncoder,
                indent=4, separators=(',', ': '), ensure_ascii=False))


class ThreadRec(ThreadPollInput):
    recv_start_index = QtCore.Signal(str, int)
    def __init__(self, name, input_stream,fid, timeout=200, flush_every_packet=False, parent=None):
        ThreadPollInput.__init__(self, input_stream, timeout=timeout, return_data=True, parent=parent)
        self.name = name
        self.fid = fid
        self._start_index = None
        self.flush_every_packet = flush_every_packet
        
    def process_data(self, pos, data):
        if self._start_index is None:
            self._start_index = int(pos - data.shape[0])
            #~ print('_start_index raw', self._start_index)
            self.recv_start_index.emit(self.name, self._start_index)
        
        #~ print(self.input_stream().name, 'pos', pos, 'data.shape', data.shape)
        self.fid.write(data.tobytes())
        
        if self.flush_every_packet:
            self.fid.flush()

register_node_type(RawRecorder)
