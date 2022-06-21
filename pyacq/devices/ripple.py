# -*- coding: utf-8 -*-
# Copyright (c) 2016, French National Center for Scientific Research (CNRS)
# Distributed under the (new) BSD License. See LICENSE for more info.

import pdb
import xipppy as xp
import numpy as np
from scipy import signal
import time
from copy import copy, deepcopy
from ..core import Node, register_node_type
from pyqtgraph.Qt import QtCore # , QtGui
from pyqtgraph.util.mutex import Mutex
from contextlib import nullcontext

ripple_analogsignal_types = ['raw', 'hi-res', 'hifreq', 'lfp']
ripple_event_types = ['stim'] # 'macro' does not have 'spk' type
ripple_signal_types = ripple_analogsignal_types + ripple_event_types
ripple_fe_types = ['macro'] # TODO: iterate front end types
_dtype_segmentDataPacket = [
    ('timestamp', 'int', (1,)), ('channel', 'int', (1,)),
    ('wf', 'int', (52,)), ('class_id', 'int', (1,))]
#
_dtype_analogsignal  = [
    ('timestamp', 'int'), ('value', 'float64')]
ripple_analogsignal_filler = np.array([(0, np.nan),], dtype=_dtype_analogsignal)

# _dtype_analogsignal = 'float64'
# ripple_analogsignal_filler = np.nan

ripple_dataReaderFuns = {
    'raw': xp.cont_raw,
    'hi-res': xp.cont_hires,
    'hifreq': xp.cont_hifreq,
    'lfp': xp.cont_lfp,
    'spk': xp.spk_data,
    'stim': xp.stim_data
    }

ripple_sample_rates = {
    'raw': int(30e3),
    'hi-res': int(2e3),
    'hifreq': int(15e3),
    'lfp': int(1e3),
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

    default_signal_type_lookup = {
        'raw': [],
        'hi-res': [eNum for eNum in range(1, 65)],
        'hifreq': [eNum for eNum in range(1, 65)],
        'lfp': [],
        }

    def __init__(
            self,
            raw_fun=None, hires_fun=None, 
            hifreq_fun=None, lfp_fun=None,
            signal_type_lookup=None):
        if raw_fun is None:
            self.raw_fun = randomSineGenerator(
                centerFreq=40, sr=ripple_sample_rates['raw'],
                noiseStd=0.05, sineAmp=1.)
        else:
            self.raw_fun = raw_fun

        if hires_fun is None:
            self.hires_fun = randomSineGenerator(
                centerFreq=40, sr=ripple_sample_rates['hi-res'],
                noiseStd=0.05, sineAmp=1.)
        else:
            self.hires_fun = hires_fun

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
        #
        self.signal_type_lookup = deepcopy(
            self.default_signal_type_lookup)
        if signal_type_lookup is not None:
            self.signal_type_lookup.update(signal_type_lookup)

    def xipppy_open(self, use_tcp=True):
        return nullcontext()

    def signal(self, chanNum, signalType):
        return chanNum in self.signal_type_lookup[signalType]

    def list_elec(self, feType):
        return [eNum for eNum in range(1, 257)]

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
            'streamtype': 'analogsignal', 'dtype': _dtype_analogsignal,
            'sample_rate': ripple_sample_rates['raw'],
            'compression': '', 'fill': ripple_analogsignal_filler},
        'hi-res': {
            'streamtype': 'analogsignal', 'dtype': _dtype_analogsignal,
            'sample_rate': ripple_sample_rates['hi-res'],
            'compression': '', 'fill': ripple_analogsignal_filler},
        'hifreq': {
            'streamtype': 'analogsignal', 'dtype': _dtype_analogsignal,
            'sample_rate': ripple_sample_rates['hifreq'],
            'compression': '', 'fill': ripple_analogsignal_filler},
        'lfp': {
            'streamtype': 'analogsignal', 'dtype': _dtype_analogsignal,
            'sample_rate': ripple_sample_rates['lfp'],
            'compression': '', 'fill': ripple_analogsignal_filler},
        'stim': {
            'streamtype': 'event', 'shape': (-1,), 'dtype': _dtype_segmentDataPacket},
        # 'spk': {
        #     'streamtype': 'event', 'dtype': _dtype_segmentDataPacket},
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
                'spk': self.xp.spk_data,
                'stim': self.xp.stim_data,
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
        self.present_analogsignal_types = []
        self.present_event_types = []
        self.present_signal_types = []
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
                        'buffer_size': int(sr * (self.sample_interval_sec + self.buffer_padding_sec)),
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


def _spikeKeyExtractor(pair):
    # pair = (chanIdx, segmentDataPacket)
    return pair[1].timestamp

class XipppyThread(QtCore.QThread):
    """
    Xipppy thread that samples data every sample_interval.
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
                # print('sleeping for {} sec'.format(max(0, next_time - time.perf_counter())))
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
                    sr = self.node.outputs[signalType].spec['sample_rate']
                    thisNumChans = self.node.outputs[signalType].spec['nb_channel']
                    nPoints = int(delta_nip_time * sr / 3e4)
                    ## [data, timestamp] = xipppy.cont_x(npoints, elecs, start_timestamp)
                    [data, timestamp] = self.node.dataReaderFuns[signalType](
                        nPoints, # read the missing number of points
                        # self.max_buffer_nip, # read as many points as you can
                        self.node.channels[signalType],
                        self.last_nip_time + 1 # start with last missing sample
                        )
                    if _dtype_analogsignal == 'float64':
                        data_out = np.array(data, dtype=_dtype_analogsignal)
                    else:
                        tOneChan = timestamp + np.arange(len(data) / thisNumChans, dtype='int') * np.round(3e4 / sr).astype('int')
                        # data = np.array([item for item in zip(t, data)], dtype=_dtype_analogsignal)
                        data_out = np.empty((len(data),), dtype=_dtype_analogsignal)
                        data_out['timestamp'] = np.tile(tOneChan, thisNumChans)
                        data_out['value'] = data
                    data_out = data_out.reshape(self.node.outputs[signalType].spec['shape'], order='F')
                    if self.node.verbose:
                        print('signal type {}\n\tread {} samples x {} chans'.format(
                            signalType, data_out.shape[0], data.shape[1]))
                        buffer_duration = data_out.shape[0] / self.node.outputs[signalType].spec['sample_rate']
                        print('\tbuffer duration: {:.3f} sec'.format(buffer_duration))
                        # print('data > 0 sum: {}'.format(np.sum((np.abs(np.asarray(data)) > 0))))
                    # self.buffers_num_samples[signalType] += data.shape[0]
                    # print('self.buffers_num_samples[{}] = {}'.format(signalType, self.buffers_num_samples[signalType]))
                    # self.node.outputs[signalType].send(data, index=self.buffers_num_samples[signalType])
                    if first_buffer:
                        self.node.outputs[signalType].spec.update({
                            't_start': timestamp / 3e4,
                            })
                    self.node.outputs[signalType].send(data_out)
                # send stim events
                sortEventOutputs = False
                for signalType in self.node.present_event_types:
                    if sortEventOutputs:
                        ########################################################
                        # Sort events by timestamp
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
                            outputPacket = np.array(
                                [
                                    (dat.timestamp, ch, np.asarray(dat.wf, dtype=int), dat.class_id,)
                                    for ch, dat in sorted(zip(chanNums, packets), key=_spikeKeyExtractor)],
                                dtype=_dtype_segmentDataPacket)
                            self.node.outputs[signalType].send(outputPacket)
                    else:
                        ########################################################
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


register_node_type(XipppyBuffer)
