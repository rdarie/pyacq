# -*- coding: utf-8 -*-
# Copyright (c) 2016, French National Center for Scientific Research (CNRS)
# Distributed under the (new) BSD License. See LICENSE for more info.

import pdb, traceback
import warnings
try:
    import xipppy
    HAVE_XIPPPY = True
    ripple_dataReaderFuns = {
        'raw': xipppy.cont_raw,
        'hi-res': xipppy.cont_hires,
        'hifreq': xipppy.cont_hifreq,
        'lfp': xipppy.cont_lfp,
        'spk': xipppy.spk_data,
        'stim': xipppy.stim_data
        }
except ImportError as error:
    warnings.warn(f"{error}")
    HAVE_XIPPPY = False
    xipppy = None
    ripple_dataReaderFuns = {
        'raw': None,
        'hi-res': None,
        'hifreq': None,
        'lfp': None,
        'spk': None,
        'stim': None
        }
from pyacq.viewers.ephyviewer_mixin import (InputStreamAnalogSignalSource, InputStreamEventAndEpochSource)
import numpy as np
from scipy import signal
import time
from copy import copy, deepcopy
from pyacq.core import Node, RPCClient, register_node_type
from pyacq.core.tools import ThreadPollInput, weakref, make_dtype
from ephyviewer.myqt import QT, QT_LIB
from pyqtgraph.util.mutex import Mutex
from contextlib import nullcontext


import array
import ctypes
import itertools

# Helper types
class DummySegmentDataPacket:
    """
    Data type to mirror classic "Segments" for spike data
    """

    def __init__(self):
        self.timestamp = 0
        self.class_id = 0
        self.wf = array.array('h', itertools.repeat(0, 52))


class DummySegmentEventPacket:
    """
    Data type to mirror classic "Segments" for digital data
    """
    def __init__(self):
        self.timestamp = 0
        self.reason = 0
        self.parallel = 0
        self.sma1 = 0
        self.sma2 = 0
        self.sma3 = 0
        self.sma4 = 0

ripple_analogsignal_types = ['raw', 'hi-res', 'hifreq', 'lfp']
ripple_event_types = ['stim'] # 'macro' does not have 'spk' type
ripple_signal_types = ripple_analogsignal_types + ripple_event_types
ripple_fe_types = ['macro'] # TODO: iterate front end types
_dtype_segmentDataPacket = [
    ('timestamp', 'int64', (1,)), ('channel', 'int', (1,)),
    ('wf', 'int', (52,)), ('class_id', 'int', (1,))]
#
_xp_spk = DummySegmentDataPacket()
ripple_event_filler = np.array([(
    _xp_spk.timestamp, 0,
    np.array(_xp_spk.wf), _xp_spk.class_id),], dtype=_dtype_segmentDataPacket)
_dtype_analogsignal  = [
    ('timestamp', 'int64'), ('value', 'float64')]
ripple_analogsignal_filler = np.array([(0, 0.),], dtype=_dtype_analogsignal)

sortEventOutputs = False
# _dtype_analogsignal = 'float64'
# ripple_analogsignal_filler = np.nan


ripple_sample_rates = {
    'raw': 30e3, # int(30e3),
    'hi-res': 2e3, # int(2e3),
    'hifreq': 15e3, # int(15e3),
    'lfp': 1e3, # int(1e3),
    }

ripple_nip_sample_periods = {
    'raw': 1, # int(30e3),
    'hi-res': 15, # int(2e3),
    'hifreq': 2, # int(15e3),
    'lfp': 30, # int(1e3),
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
        data = (noiseWave + sineWave).astype('float64').reshape(-1, order='F')
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
        sineWave = sineAmp * np.asarray(signal.chirp(t_adj, startFreq, freqPeriod, stopFreq))[:, None]
        data = (noiseWave + sineWave).astype('float32').reshape(-1, order='F')
        # print('RandomChirp: data.shape = {}'.format(data.shape))
        timestamp = start_timestamp
        return data, timestamp
    return dummyRandomChirp

rng = np.random.default_rng(12345)

def dummySpk(t_start, t_stop, max_spk):
    t_interval = t_stop - t_start
    spkFreq = 5 # Hz
    count = np.round((spkFreq * t_interval / 3e4) * (1 + (rng.random() - 0.5) / 10.)).astype('int')
    if count == 0:
        return 0, []
    timestamps = rng.random((count+2,))
    timestamps /= timestamps.sum()
    timestamps = np.floor(timestamps.cumsum() * t_interval)
    # timestamps = np.linspace(0, t_interval * (1 - rng.random() / 10.), count)
    data = []
    for dt in timestamps[1:-1]:
        if HAVE_XIPPPY:
            spk = xipppy.SegmentDataPacket()
        else:
            spk = DummySegmentDataPacket()
        spk.timestamp = t_stop - dt
        data.append(spk)
    return count, data

class DummyXipppy():
    t_zero = time.time()
    _num_elecs = 16

    SegmentDataPacket = DummySegmentDataPacket

    default_signal_type_lookup = {
        'raw': [],
        'hi-res': [eNum for eNum in range(1, _num_elecs + 1)],
        'hifreq': [eNum for eNum in range(1, _num_elecs + 1)],
        'lfp': [],
        'stim': [eNum for eNum in range(1, _num_elecs + 1)],
        }

    def __init__(
            self,
            raw_fun=None, hires_fun=None, 
            hifreq_fun=None, lfp_fun=None,
            stim_spk_fun=None,
            signal_type_lookup=None):
        #
        if raw_fun is None:
            self.raw_fun = randomSineGenerator(
                centerFreq=40, sr=ripple_sample_rates['raw'],
                noiseStd=0.05, sineAmp=1.)
        else:
            self.raw_fun = raw_fun
        #
        if hires_fun is None:
            self.hires_fun = randomSineGenerator(
                centerFreq=40, sr=ripple_sample_rates['hi-res'],
                noiseStd=0.05, sineAmp=1.)
        else:
            self.hires_fun = hires_fun
        #
        if hifreq_fun is None:
            self.hifreq_fun = randomChirpGenerator(
                startFreq=10, stopFreq=40, freqPeriod=2.,
                sr=ripple_sample_rates['hifreq'], noiseStd=0.05, sineAmp=1.)
        else:
            self.hifreq_fun = hifreq_fun

        if lfp_fun is None:
            self.lfp_fun = randomSineGenerator(
                centerFreq=40, sr=ripple_sample_rates['lfp'],
                noiseStd=0.05, sineAmp=1.)
        else:
            self.lfp_fun = lfp_fun
        
        _t_now = self.time()
        self.last_stim_spk_t = {ch: _t_now for ch in self.list_elec()}

        if stim_spk_fun is None:
            self.stim_spk_fun = dummySpk
        else:
            self.stim_spk_fun = stim_spk_fun
        #
        self.signal_type_lookup = deepcopy(
            self.default_signal_type_lookup)
        if signal_type_lookup is not None:
            self.signal_type_lookup.update(signal_type_lookup)

    def xipppy_open(self, use_tcp=True):
        return nullcontext()

    def signal(self, chanNum, signalType):
        return chanNum in self.signal_type_lookup[signalType]

    def list_elec(self, feType=None):
        return [eNum for eNum in range(1, self._num_elecs + 1)]

    def _close(self):
        return

    def time(self):
        t = int((time.time() - self.t_zero) * 3e4)
        return t

    def cont_raw(self, npoints, elecs, start_timestamp):
        return self.raw_fun(npoints, elecs, start_timestamp)

    def cont_hires(self, npoints, elecs, start_timestamp):
        return self.hires_fun(npoints, elecs, start_timestamp)
        
    def cont_hifreq(self, npoints, elecs, start_timestamp):
        return self.hifreq_fun(npoints, elecs, start_timestamp)
        
    def cont_lfp(self, npoints, elecs, start_timestamp):
        return self.lfp_fun(npoints, elecs, start_timestamp)

    def stim_data(self, elecs, max_spk):
        t_now = self.time()
        count, data = self.stim_spk_fun(self.last_stim_spk_t[elecs], t_now, max_spk)
        self.last_stim_spk_t[elecs] = t_now
        return count, data

class XipppyTxBuffer(Node):
    """
    A buffer for data streamed from a Ripple NIP via xipppy.
    """
    _sortEventOutputs = sortEventOutputs
    _output_specs = {
        signalType: {
            'streamtype': 'analogsignal', 'dtype': _dtype_analogsignal,
            'sample_rate': ripple_sample_rates[signalType],
            'nip_sample_period': ripple_nip_sample_periods[signalType],
            'compression': '', 'fill': ripple_analogsignal_filler}
        for signalType in ripple_analogsignal_types
        }
    _output_specs.update({
        'stim': {
            'streamtype': 'event', 'shape': (-1,), 'sorted_by_time': _sortEventOutputs,
            'fill': ripple_event_filler, 'buffer_size': 10000, 'dtype': _dtype_segmentDataPacket},
        # 'spk': {
        #     'streamtype': 'event', 'dtype': _dtype_segmentDataPacket},
            })

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
                # 'spk': self.xp.spk_data,
                'stim': self.xp.stim_data,
                }
        else:
            self.dataReaderFuns = ripple_dataReaderFuns
            self.xp = xipppy
        #
        self.verbose = False
        #
        self.channels = {}
        self.sample_interval_sec = None
        self.sample_interval_nip = None
        self.sample_chunksize_sec = None
        self.buffer_size_sec = None
        self.latency_padding_sec = None
        self.allElecs = []
        self.present_analogsignal_types = []
        self.present_event_types = []
        self.present_signal_types = []
        self.thread = None
        self.xipppy_use_tcp = True

    def _configure(
            self,
            sample_interval_sec=1., sample_chunksize_sec=.5,
            buffer_size_sec=250e-3, latency_padding_sec=50e-3,
            xipppy_use_tcp=True,
            channels={}, verbose=False, debugging=False):
        #
        self.xipppy_use_tcp = xipppy_use_tcp
        self.sample_interval_sec = sample_interval_sec
        self.sample_interval_nip = int(self.sample_interval_sec * 3e4)
        self.sample_chunksize_sec = sample_chunksize_sec
        self.buffer_size_sec = buffer_size_sec
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
                    print('XipppyTxBuffer; configure(); Signal type {}'.format(signalType))
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
                if signalType in ripple_analogsignal_types:
                    self.channels[signalType] = [
                        chanNum
                        for chanNum in presentChannels
                        if self.xp.signal(chanNum, signalType)]
                elif signalType in ripple_event_types:
                    self.channels[signalType] = [
                        chanNum
                        for chanNum in presentChannels]
                # update nb of channels
                thisNumChans = len(self.channels[signalType])
                if self.verbose:
                    print('Signal type {}, {} channels found.'.format(signalType, thisNumChans))
                self.outputs[signalType].spec.update({
                    'nb_channel': thisNumChans,
                    })
                if signalType in ripple_analogsignal_types:
                    sr = self.outputs[signalType].spec['sample_rate']
                    self.outputs[signalType].spec.update({
                        'chunksize': int(sr * self.sample_chunksize_sec),
                        'buffer_size': int(sr * (self.sample_interval_sec + self.buffer_size_sec)),
                        })
                    if thisNumChans > 0:
                        self.outputs[signalType].spec.update({
                            'shape': (-1, thisNumChans),
                            })
                        if self.verbose:
                            print(
                                "Setting self.outputs['{}'].spec['shape'] = {}".format(
                                    signalType, (-1, thisNumChans) ))
                        self.present_analogsignal_types.append(signalType)
                elif signalType in ripple_event_types:
                    if thisNumChans > 0:
                        self.present_event_types.append(signalType)
            self.present_signal_types = self.present_analogsignal_types + self.present_event_types
    
    def after_output_configure(self, signalType):
        channel_info = [
            {
                'channel_index': c,
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


class XipppyRxBuffer(Node):
    """
    A buffer for data streamed from a Ripple NIP via xipppy.
    """
    _output_specs = {
        signalType: {
            'streamtype': 'analogsignal', 'dtype': _dtype_analogsignal,
            'sample_rate': ripple_sample_rates[signalType],
            'nip_sample_period': ripple_nip_sample_periods[signalType],
            'compression': '', 'fill': ripple_analogsignal_filler}
        for signalType in ripple_analogsignal_types
        }
    _output_specs.update({
        'stim': {
            'streamtype': 'event', 'shape': (-1,), 'sorted_by_time': sortEventOutputs,
            'fill': ripple_event_filler, 'buffer_size': 10000, 'dtype': _dtype_segmentDataPacket},
        # 'spk': {
        #     'streamtype': 'event', 'dtype': _dtype_segmentDataPacket},
            })
    _input_specs = {
        signalType: {
            'streamtype': 'analogsignal', 'dtype': _dtype_analogsignal,
            'sample_rate': ripple_sample_rates[signalType],
            'nip_sample_period': ripple_nip_sample_periods[signalType],
            'fill': ripple_analogsignal_filler}
        for signalType in ripple_analogsignal_types
        }
    _input_specs.update({
        'stim': {
            'streamtype': 'event', 'shape': (-1,), 'sorted_by_time': sortEventOutputs,
            'fill': ripple_event_filler, 'dtype': _dtype_segmentDataPacket},
        # 'spk': {
        #     'streamtype': 'event', 'dtype': _dtype_segmentDataPacket},
            })

    def __init__(
            self, requested_signal_types=None,
            **kargs):
        if requested_signal_types is None:
            self.requested_signal_types = ripple_signal_types
        else:
            self.requested_signal_types = requested_signal_types
        self.requested_analogsignal_types = [
            sig_type for sig_type in self.requested_signal_types
            if sig_type in ripple_analogsignal_types
            ]
        self.requested_event_types = [
            sig_type for sig_type in self.requested_signal_types
            if sig_type in ripple_event_types
            ]
        self.nb_channel = {}
        self.sources = {}
        self.pollers = {}
        self.source_threads = {}
        Node.__init__(self, **kargs)
        
    def _configure(self):
        pass

    def _check_nb_channel(self):
        for inputname in self.requested_signal_types:
            self.nb_channel[inputname] = self.inputs[inputname].params['nb_channel']
            # print('self.nb_channel[{}] = {}'.format(inputname, self.nb_channel[inputname]))

    def _initialize(self):
        self._check_nb_channel()
        for inputname in self.requested_analogsignal_types:
            if self.nb_channel[inputname] > 0:
                # set_buffer(self, size=None, double=True, axisorder=None, shmem=None, fill=None)
                bufferParams = {
                    key: self.inputs[inputname].params[key] for key in ['double', 'axisorder', 'fill']}
                bufferParams['size'] = self.inputs[inputname].params['buffer_size']
                if (self.inputs[inputname].params['transfermode'] == 'sharedmem'):
                    if 'shm_id' in self.inputs[inputname].params:
                        bufferParams['shmem'] = self.inputs[inputname].params['shm_id']
                    else:
                        bufferParams['shmem'] = True
                else:
                    bufferParams['shmem'] = None
                self.inputs[inputname].set_buffer(**bufferParams)
                self.sources[inputname] = InputStreamAnalogSignalSource(self.inputs[inputname])
                #
                thread = QT.QThread()
                self.source_threads[inputname] = thread
                self.sources[inputname].moveToThread(thread)
                # 
                self.pollers[inputname] = ThreadPollInput(self.inputs[inputname])
                # self.pollers[inputname].new_data.connect(self.analogsignal_received)
        for inputname in self.requested_event_types:
            # no need for a buffer, will split by channel
            self.sources[inputname] = InputStreamEventAndEpochSource(self.inputs[inputname])
            thread = QT.QThread()
            self.source_threads[inputname] = thread
            self.sources[inputname].moveToThread(thread)
            self.pollers[inputname] = ThreadPollInput(self.inputs[inputname], return_data=True)
            self.pollers[inputname].new_data.connect(self.sources[inputname].event_received)
    
    def _start(self):
        for inputname, poller in self.pollers.items():
            poller.start()
            self.source_threads[inputname].start()

    def _stop(self):
        for inputname, poller in self.pollers.items():
            poller.stop()
            poller.wait()
            self.source_threads[inputname].stop()
            self.source_threads[inputname].wait()
    
    def _close(self):
        for inputname in self.requested_event_types:
            source = self.sources[inputname]
            for buffer in source.buffers_by_channel:
                if buffer.shm_id is not None:
                    self.buffer._shm.close()
        if self.running():
            self.stop()

    def analogsignal_received(self, ptr, data):
        print(f"Analog signal data received: {ptr} {data}")
        return


def _spikeKeyExtractor(pair):
    # pair = (chanIdx, segmentDataPacket)
    return pair[1].timestamp

class XipppyThread(QT.QThread):
    """
    Xipppy thread that samples data every sample_interval.
    """
    def __init__(self, node, parent=None):
        QT.QThread.__init__(self, parent=parent)
        #
        self.lock = Mutex()
        self.running = False
        # xipppy 5 second buffer duration
        self.max_buffer_nip = int(5 * 3e4)
        #
        self.head = 0
        self.last_nip_time = 0
        # self.buffers = {}
        # self.buffers_num_samples = {}
        self.num_requests = 0
        #
        self.node = node

    def run(self):
        with self.lock:
            self.running = True
        interval = self.node.sample_interval_sec
        with self.node.xp.xipppy_open(use_tcp=self.node.xipppy_use_tcp):
            first_buffer = True
            next_time = time.perf_counter() + interval
            while True:
                with self.lock:
                    if not self.running:
                        break
                # print('XipppyThread: sleeping for {:.3f} sec'.format(max(0, next_time - time.perf_counter())))
                time.sleep(max(0, next_time - time.perf_counter()))
                # get current nip time
                self.head = self.node.xp.time()
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
                for signalType in self.node.present_analogsignal_types:
                    # if first_buffer:
                    #     self.buffers_num_samples[signalType] = 0
                    points_per_period = self.node.outputs[signalType].spec['nip_sample_period']
                    thisNumChans = self.node.outputs[signalType].spec['nb_channel']
                    nPoints = int(delta_nip_time / points_per_period)
                    ## [data, timestamp] = xipppy.cont_x(npoints, elecs, start_timestamp)
                    [data, timestamp] = self.node.dataReaderFuns[signalType](
                        nPoints, # read the missing number of points
                        # self.max_buffer_nip, # read as many points as you can
                        self.node.channels[signalType],
                        self.last_nip_time # start with last missing sample
                        )
                    if _dtype_analogsignal == 'float64':
                        data_out = np.array(data, dtype=_dtype_analogsignal)
                    else:
                        tOneChan = timestamp + np.arange(len(data) / thisNumChans, dtype='int64') * points_per_period
                        # data = np.array([item for item in zip(t, data)], dtype=_dtype_analogsignal)
                        data_out = np.empty((len(data),), dtype=_dtype_analogsignal)
                        data_out['timestamp'] = np.tile(tOneChan, thisNumChans)
                        data_out['value'] = data
                    data_out = data_out.reshape(
                        self.node.outputs[signalType].spec['shape'], order='F')
                    if self.node.verbose:
                        print('signal type {}\n\tread {} samples x {} chans'.format(
                            signalType, data_out.shape[0], data.shape[1]))
                        buffer_duration = data_out.shape[0] / self.node.outputs[signalType].spec['sample_rate']
                        print('\tbuffer duration: {:.3f} sec'.format(buffer_duration))
                        # print('data > 0 sum: {}'.format(np.sum((np.abs(np.asarray(data)) > 0))))
                    # self.buffers_num_samples[signalType] += data.shape[0]
                    # print('self.buffers_num_samples[{}] = {}'.format(signalType, self.buffers_num_samples[signalType]))
                    # self.node.outputs[signalType].send(data, index=self.buffers_num_samples[signalType])
                    '''
                    if first_buffer:
                        self.node.outputs[signalType].spec.update({
                            't_start': timestamp / 3e4,
                            })
                            '''
                    equiv_index = int(timestamp / points_per_period + data_out.shape[0])
                    # print(f'{signalType}: equiv_index = {equiv_index}')
                    self.node.outputs[signalType].send(data_out, index = equiv_index)
                # send stim events
                for signalType in self.node.present_event_types:
                    '''
                    if self.node._sortEventOutputs:
                        '''
                    chanNums = []
                    packets = []
                    for chanIdx in self.node.channels[signalType]:
                        ## (count, data) = xipppy.stim_data(elecs, max_spk)
                        [_, packetList] = self.node.dataReaderFuns[signalType](chanIdx, 1023)
                        chanNums += [chanIdx for _ in packetList]
                        packets += packetList
                        if len(packetList) and self.node.verbose:
                            print([(dat.timestamp, chanIdx, dat.class_id) for dat in packetList])
                    nPackets = len(packets)
                    if nPackets > 0:
                        if self.node._sortEventOutputs:
                            ########################################################
                            # Sort events by timestamp
                            outputPacket = np.array(
                                [
                                    (dat.timestamp, ch, np.asarray(dat.wf, dtype=int), dat.class_id,)
                                    for ch, dat in sorted(zip(chanNums, packets), key=_spikeKeyExtractor)],
                                dtype=_dtype_segmentDataPacket)
                        else:
                            ########################################################
                            # Leave items sorted by channel
                            outputPacket = np.array(
                                [
                                    (dat.timestamp, ch, np.asarray(dat.wf, dtype=int), dat.class_id,)
                                    for ch, dat in zip(chanNums, packets)],
                                dtype=_dtype_segmentDataPacket)
                        self.node.outputs[signalType].send(outputPacket)
                    '''
                    else:
                        ########################################################
                        # send each channel's data individually
                        # do not sort event outputs
                        for chanIdx in self.node.channels[signalType]:
                            ## (count, data) = xipppy.stim_data(elecs, max_spk)
                            [_, packetList] = self.node.dataReaderFuns[signalType](chanIdx, 1023)
                            nPackets = len(packetList)
                            if nPackets > 0:
                                outputPacket = np.array(
                                    [
                                        (dat.timestamp, chanIdx, np.asarray(dat.wf, dtype=int), dat.class_id,)
                                        for dat in packetList],
                                    dtype=_dtype_segmentDataPacket)
                                self.node.outputs[signalType].send(outputPacket)
                    '''
                first_buffer = False
                self.num_requests += 1
                self.last_nip_time = copy(self.head)
                if self.node.debugging:
                    if self.num_requests > 3:
                        self.running = False
                next_time += (time.perf_counter() - next_time) // interval * interval + interval

    def stop(self):
        with self.lock:
            self.running = False


register_node_type(XipppyTxBuffer)


class RippleThreadStreamConverter(ThreadPollInput):
    """Thread that polls for data on an input stream and converts the transfer
    mode or time axis of the data before relaying it through its output.
    """
    def __init__(
            self, input_stream, output_stream,
            signal_type=None, nip_sample_period=None,
            timeout=200, parent=None):
        ThreadPollInput.__init__(
            self, input_stream, timeout=timeout,
            return_data=True, parent=parent)
        self.output_stream = weakref.ref(output_stream)
        self.signal_type = signal_type
        self.nip_sample_period = nip_sample_period
        self.output_dtype = make_dtype(self.output_stream().params['dtype'])
        
    def process_data(self, pos, data):
        if self.signal_type == 'analog':
            output = data['value']
            self.output_stream().send(output, index=pos)
        elif self.signal_type == 'events':
            mask = data['channel'] == 13
            output = np.sort(data['timestamp'][mask] / self.nip_sample_period).astype(self.output_dtype)
            # print(f"stream convert trig {output}")
            self.output_stream().send(output, index=pos)


class RippleStreamAdapter(Node):

    _input_specs = {
        'signals': {
            'streamtype': 'analogsignal', 'dtype': _dtype_analogsignal,
            'compression': '', 'fill': ripple_analogsignal_filler},
        'events' : {
            'streamtype': 'event', 'shape': (-1,),
            'fill': ripple_event_filler, 'dtype': _dtype_segmentDataPacket},
        }
    _output_specs = {
        'signals': {
            'streamtype': 'analogsignal', 'dtype': 'float64',
            'compression': '', 'fill': 0., },
        'events' : {
            'streamtype': 'event', 'shape': (-1,),
            'fill': ripple_event_filler, 'dtype': 'int64',}, 
        }
    
    def __init__(self, **kargs):
        Node.__init__(self, **kargs)
        self.threads = {}
    
    def _configure(self):
        pass

    def _initialize(self):
        self.sample_rate = self.inputs['signals'].params['sample_rate']
        self.nip_sample_period  = self.inputs['signals'].params['nip_sample_period']
        for param_name in ['sample_rate', 'nip_sample_period', 'shape']:
            self.outputs['signals'].params[param_name] = self.inputs['signals'].params[param_name]
        #
        self.threads['signals'] = RippleThreadStreamConverter(
            self.inputs['signals'], self.outputs['signals'],
            signal_type='analog', nip_sample_period=self.nip_sample_period)
        self.threads['events'] = RippleThreadStreamConverter(
            self.inputs['events'], self.outputs['events'],
            signal_type='events', nip_sample_period=self.nip_sample_period)
        # print(f"self.outputs['signals'].params['shape'] = {self.outputs['signals'].params['shape']}")
        # print(f"self.outputs['events'].params.keys() = {self.outputs['events'].params.keys()}")
        # print(f"self.inputs['signals'].params['shape'] = {self.inputs['signals'].params['shape']}")
        # print(f"self.inputs['events'].params.keys() = {self.inputs['events'].params.keys()}")
    
    def _start(self):
        self.threads['signals'].start()
        self.threads['events'].start()

    def _stop(self):
        self.threads['signals'].stop()
        self.threads['signals'].wait()
        self.threads['events'].stop()
        self.threads['events'].wait()
    
    def _close(self):
        pass

register_node_type(RippleStreamAdapter)

class XippyServerWindow(QT.QMainWindow):
    '''  '''

    def __init__(
            self, server=None, parent=None):
        QT.QMainWindow.__init__(self, parent)
        self.server = server
        self.client = None
        self.setWindowTitle('xipppy server')

    def start(self):
        self.server.run_forever()
        self.client = RPCClient.get_client(self.server.address)

    def closeEvent(self, event):
        if self.client is not None:
            self.client.close_server()
        event.accept()