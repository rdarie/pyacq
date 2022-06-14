# -*- coding: utf-8 -*-
# Copyright (c) 2016, French National Center for Scientific Research (CNRS)
# Distributed under the (new) BSD License. See LICENSE for more info.

import pdb
import xipppy as xp
import numpy as np
from scipy import signal
import time
from copy import copy

from ..core import Node, register_node_type
from pyqtgraph.Qt import QtCore # , QtGui
from pyqtgraph.util.mutex import Mutex
from contextlib import nullcontext

ripple_signal_types = ['raw', 'hi-res', 'hifreq', 'lfp']
ripple_fe_types = ['macro'] # TODO: iterate front end types

ripple_dataReaderFuns = {
    'raw': xp.cont_raw,
    'hi-res': xp.cont_hires,
    'hifreq': xp.cont_hifreq,
    'lfp': xp.cont_lfp,
    }

# dummy input
def dummyRandom(
        npoints=None, elecs=None,
        start_timestamp=0):
    ## [data, timestamp] = xipppy.cont_x(npoints, elecs, start_timestamp)
    nb_channel = len(elecs)
    data = np.random.normal(
        size=(1, npoints * nb_channel)).astype('float32')
    timestamp = start_timestamp
    return data, timestamp


def randomSineGenerator(
        centerFreq=20, sr=1000,
        noiseStd=1., sineAmp=1.):
    #
    def dummyRandomSine(
            npoints=None, elecs=None,
            start_timestamp=0):
        ## [data, timestamp] = xipppy.cont_x(npoints, elecs, start_timestamp)
        nb_channel = len(elecs)
        noiseWave = noiseStd * np.random.normal(size=(npoints, nb_channel))
        #
        t_start = start_timestamp / 3e4
        t = t_start + np.arange(npoints) / sr
        sineWave = sineAmp * np.sin(2 * np.pi * centerFreq * t)[:, None]
        #
        data = (noiseWave + sineWave).astype('float32').reshape(1, -1, order='F')
        # print('randomSine: data.shape = {}'.format(data.shape))
        timestamp = start_timestamp
        return data, timestamp
    return dummyRandomSine


def randomChirpGenerator(
        startFreq=10, stopFreq=40, freqPeriod=2.,
        sr=1000, noiseStd=1., sineAmp=1.):
    #
    def dummyRandomChirp(
            npoints=None, elecs=None,
            start_timestamp=0):
        ## [data, timestamp] = xipppy.cont_x(npoints, elecs, start_timestamp)
        nb_channel = len(elecs)
        noiseWave = noiseStd * np.random.normal(size=(npoints, nb_channel))
        t_start = start_timestamp / 3e4
        t = t_start + np.arange(npoints) / sr
        t_adj = (1. + signal.sawtooth(2 * np.pi * t / freqPeriod)) * freqPeriod / 2
        # pdb.set_trace()
        sineWave = sineAmp * np.asarray(signal.chirp(t_adj, startFreq, freqPeriod, stopFreq))[:, None]
        data = (noiseWave + sineWave).astype('float32').reshape(1, -1, order='F')
        # print('RandomChirp: data.shape = {}'.format(data.shape))
        timestamp = start_timestamp
        return data, timestamp
    return dummyRandomChirp


class DummyXipppy():
    def __init__(
            self,
            raw_fun=None, hires_fun=None, hifreq_fun=None, lfp_fun=None):
        if raw_fun is None:
            self.raw_fun = dummyRandom
        else:
            self.raw_fun = raw_fun

        if hires_fun is None:
            self.hires_fun = dummyRandom
        else:
            self.hires_fun = hires_fun

        if hifreq_fun is None:
            self.hifreq_fun = dummyRandom
        else:
            self.hifreq_fun = hifreq_fun

        if lfp_fun is None:
            self.lfp_fun = dummyRandom
        else:
            self.lfp_fun = lfp_fun
        #
        self.signal_type_lookup = {
            'raw': [],
            'hi-res': [eNum for eNum in range(1, 65)],
            'hifreq': [eNum for eNum in range(1, 65)],
            'lfp': [],
            }


    def xipppy_open(self, use_tcp=True):
        return nullcontext()

    def signal(self, chanNum, signalType):
        return chanNum in self.signal_type_lookup[signalType]

    def list_elec(self, feType):
        return [eNum for eNum in range(1, 65)]

    def _close(self):
        return

    def time(self):
        t = int((time.time() - 1655100000.) * 3e4)
        return t

    def cont_raw(self, npoints, elecs, start_timestamp):
        return self.raw_fun(npoints, elecs, start_timestamp)

    def cont_hires(self, npoints, elecs, start_timestamp):
        return self.hires_fun(npoints, elecs, start_timestamp)
        
    def cont_hifreq(self, npoints, elecs, start_timestamp):
        return self.hifreq_fun(npoints, elecs, start_timestamp)
        
    def cont_lfp(self, npoints, elecs, start_timestamp):
        return self.lfp_fun(npoints, elecs, start_timestamp)

class XipppyBuffer(Node):
    """
    A buffer for data streamed from a Ripple NIP via xipppy.
    """

    _output_specs = {
        'raw': {
            'streamtype': 'analogsignal', 'dtype': 'float32',
            'sample_rate': int(30e3), 'compression': ''},
        'hi-res': {
            'streamtype': 'analogsignal', 'dtype': 'float32',
            'sample_rate': int(2e3), 'compression': ''},
        'hifreq': {
            'streamtype': 'analogsignal', 'dtype': 'float32',
            'sample_rate': int(15e3), 'compression': ''},
        'lfp': {
            'streamtype': 'analogsignal', 'dtype': 'float32',
            'sample_rate': int(1e3), 'compression': ''},
        }
    

    def __init__(self, dummy=False, dummy_kwargs=dict(), **kargs):
        Node.__init__(self, **kargs)
        self.dummy = dummy
        if self.dummy:
            self.xp = DummyXipppy(**dummy_kwargs)
            self.dataReaderFuns = {
                'raw': self.xp.cont_raw,
                'hi-res': self.xp.cont_hires,
                'hifreq': self.xp.cont_hifreq,
                'lfp': self.xp.cont_lfp,
                }
        else:
            self.dataReaderFuns = ripple_dataReaderFuns
            self.xp = xp
        #
        self.verbose = False
        #
        self.channels = {}
        self.sample_interval_sec = None
        self.sample_interval_nip = None
        self.sample_chunksize_sec = None
        self.buffer_padding_sec = None
        self.latency_padding_sec = None
        self.allElecs = []
        self.thread = None
        self.xipppy_use_tcp = True

    def _configure(
            self,
            sample_interval_sec=1., sample_chunksize_sec=.5,
            buffer_padding_sec=250e-3, latency_padding_sec=50e-3,
            xipppy_use_tcp=True,
            channels={}, verbose=False, debugging=False):
        #
        self.xipppy_use_tcp = xipppy_use_tcp
        self.sample_interval_sec = sample_interval_sec
        self.sample_interval_nip = int(self.sample_interval_sec * 3e4)
        self.sample_chunksize_sec = sample_chunksize_sec
        self.buffer_padding_sec = buffer_padding_sec
        self.latency_padding_sec = latency_padding_sec
        self.verbose = verbose
        self.debugging = debugging
        #
        if self.verbose:
            print('self.sample_interval_nip = {}'.format(self.sample_interval_nip))
        #
        with self.xp.xipppy_open(use_tcp=self.xipppy_use_tcp):
            # get list of channels that actually exist
            self.allElecs = self.xp.list_elec('macro')
            #
            for signalType in ripple_signal_types:
                # configure list of channels to stream
                if signalType in channels:
                    self.channels[signalType] = channels[signalType]
                else:
                    self.channels[signalType] = []
                if self.verbose:
                    print('XipppyBuffer; configure(); Signal type {}'.format(signalType))
                # prune list of channels
                if len(self.channels[signalType]):
                    # we requested a specific list
                    presentChannels = [
                        chanNum
                        for chanNum in self.channels[signalType]
                        if (chanNum in self.allElecs)
                        ]
                else:
                    presentChannels = self.allElecs
                self.channels[signalType] = [
                    chanNum
                    for chanNum in presentChannels
                    if self.xp.signal(chanNum, signalType)]
                # update nb of channels
                thisNumChans = len(self.channels[signalType])
                sr = self.outputs[signalType].spec['sample_rate']
                self.outputs[signalType].spec.update({
                    'nb_channel': thisNumChans,
                    'chunksize': int(sr * self.sample_chunksize_sec),
                    'buffer_size': int(sr * (self.sample_interval_sec + self.buffer_padding_sec)),
                    })
                #
                if self.verbose:
                    print('Signal type {}, {} channels found.'.format(signalType, thisNumChans))
                if thisNumChans > 0:
                    #
                    self.outputs[signalType].spec.update({
                        'shape': (-1, thisNumChans),
                        })
                    if self.verbose:
                        print(
                            "Setting self.outputs['{}'].spec['shape'] = {}".format(
                                signalType, (-1, thisNumChans) ))
    
    def after_output_configure(self, signalType):
        channel_info = [
            {
                'name': '{}_{}'.format(signalType, c)}
            for c in self.channels[signalType]]
        self.outputs[signalType].params['channel_info'] = channel_info

    def _initialize(self):
        self.thread = XipppyThread(self, parent=None)

    def _start(self):
        self.thread.start()

    def _stop(self):
        self.thread.stop()
        self.thread.wait()
    
    def _close(self):
        self.xp._close()


class XipppyThread(QtCore.QThread):
    """
    Xipppy thread that continuously data.
    """
    def __init__(self, node, parent=None):
        QtCore.QThread.__init__(self, parent=parent)
        #
        self.lock = Mutex()
        self.running = False
        # xipppy 5 second buffer duration
        self.max_buffer_nip = int(5 * 3e4)
        #
        self.head = 0
        self.last_nip_time = 0
        self.buffers = {}
        self.buffers_num_samples = {}
        self.num_requests = 0
        #
        self.node = node

    def run(self):
        with self.lock:
            self.running = True
        #
        with self.node.xp.xipppy_open(use_tcp=self.node.xipppy_use_tcp):
            first_buffer = True
            while True:
                with self.lock:
                    if not self.running:
                        break
                t_start_loop = time.perf_counter()
                # get current nip time
                self.head = self.node.xp.time()
                time.sleep(self.node.latency_padding_sec)
                if first_buffer:
                    self.last_nip_time = self.head - self.node.sample_interval_nip
                # delta_nip_time: actual elapsed ripple ticks since last read
                delta_nip_time = self.head - self.last_nip_time
                if self.node.verbose:
                    print('XipppyThread.run()')
                    print('\t          head  = {}'.format(self.head))
                    print('\t                  {:.3f} sec'.format(self.head / 3e4))
                    print('\t last_nip_time  = {}'.format(self.last_nip_time))
                    print('\t                  {:.3f} sec'.format(self.last_nip_time / 3e4))
                    # print('\t delta_nip_time = {}'.format(delta_nip_time))
                    # print('\t delta_sec      = {} sec'.format(delta_nip_time / 3e4))
                # check that we haven't exhausted the 5 sec xipppy buffer
                if delta_nip_time > self.max_buffer_nip:
                    self.last_nip_time = self.head - self.max_buffer_nip
                    delta_nip_time = self.head - self.last_nip_time
                    if self.node.verbose:
                        print('Warning! self.last_nip_time is more than 5 sec in the past')
                #
                for signalType in ripple_signal_types:
                    thisNumChans = self.node.outputs[signalType].spec['nb_channel']
                    if first_buffer:
                        self.buffers_num_samples[signalType] = 0
                    if thisNumChans > 0:
                        nPoints = int(
                            delta_nip_time *
                            self.node.outputs[signalType].spec['sample_rate'] / 3e4)
                        ## [data, timestamp] = xipppy.cont_x(npoints, elecs, start_timestamp)
                        [data, _] = self.node.dataReaderFuns[signalType](
                            nPoints, # read the missing number of points
                            # self.max_buffer_nip, # read as many points as you can
                            self.node.channels[signalType],
                            self.last_nip_time + 1 # start with last missing sample
                            )
                        self.buffers[signalType] = np.reshape(
                            np.asarray(data), self.node.outputs[signalType].spec['shape'], order='F')
                        if self.node.verbose:
                            print('signal type {}\n\tread {} samples x {} chans'.format(
                                signalType, self.buffers[signalType].shape[0], self.buffers[signalType].shape[1]))
                            buffer_duration = self.buffers[signalType].shape[0] / self.node.outputs[signalType].spec['sample_rate']
                            print('\tbuffer duration: {:.3f} sec'.format(buffer_duration))
                            # print('data > 0 sum: {}'.format(np.sum((np.abs(np.asarray(data)) > 0))))
                        self.buffers_num_samples[signalType] += self.buffers[signalType].shape[0]
                        self.node.outputs[signalType].send(
                            self.buffers[signalType],
                            index=self.buffers_num_samples[signalType]
                            )
                #
                first_buffer = False
                self.num_requests += 1
                self.last_nip_time = copy(self.head)
                if self.node.debugging:
                    if self.num_requests > 3:
                        self.running = False
                t_sleep = max(0., self.node.sample_interval_sec - time.perf_counter() + t_start_loop)
                time.sleep(t_sleep)

    def stop(self):
        with self.lock:
            self.running = False

register_node_type(XipppyBuffer)
