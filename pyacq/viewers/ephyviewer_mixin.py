# -*- coding: utf-8 -*-
#~ from __future__ import (unicode_literals, print_function, division, absolute_import)

from tkinter import N
import numpy as np

from collections import OrderedDict
from ephyviewer.myqt import QT, QT_LIB
import pyqtgraph as pg
from pyqtgraph.util.mutex import Mutex
import logging
import time
import atexit
import weakref
import sys
import pdb
from ephyviewer.datasource import BaseAnalogSignalSource, BaseEventAndEpoch, BaseSpikeSource

from ephyviewer.mainviewer import MainViewer, location_to_qt, orientation_to_qt
from ephyviewer.navigation import NavigationToolBar

from pyacq.core import (Node, ThreadPollInput, RingBuffer)
from pyacq.devices.ripple import (
    ripple_nip_sample_periods, ripple_analogsignal_filler, sortEventOutputs,
    ripple_analogsignal_types, ripple_event_types, ripple_signal_types, ripple_sample_rates,
    _dtype_analogsignal, _dtype_segmentDataPacket, ripple_event_filler)

import sklearn.metrics

from pyacq.dsp.triggeraccumulator import TriggerAccumulator, ThreadPollInputUntilPosLimit

from tridesclous.waveformtools import extract_chunks
from tridesclous import labelcodes
from tridesclous.gui.base import ControllerBase
from tridesclous.gui.traceviewer import CatalogueTraceViewer
from tridesclous.gui.onlinepeaklists import OnlinePeakList, OnlineClusterPeakList
from tridesclous.gui.onlinewaveformviewer import RippleWaveformViewer
from tridesclous.gui.pairlist import PairList
from tridesclous.gui.waveformhistviewer import WaveformHistViewer

from tridesclous.tools import (median_mad, mean_std, make_color_dict, get_color_palette)

_dtype_peak = [
    ('index', 'int64'), ('cluster_label', 'int64'), ('channel', 'int64'),
    ('segment', 'int64'), ('extremum_amplitude', 'float64'), ('timestamp', 'float64'),]
_dtype_peak_zero = np.zeros((1,), dtype=_dtype_peak)
_dtype_peak_zero['cluster_label'] = labelcodes.LABEL_UNCLASSIFIED
_dtype_cluster = [
    ('cluster_label', 'int64'), ('cell_label', 'int64'), 
    ('extremum_channel', 'int64'), ('extremum_amplitude', 'float64'),
    ('waveform_rms', 'float64'), ('nb_peak', 'int64'), 
    ('tag', 'U16'), ('annotations', 'U32'), ('color', 'uint32')]


LOGGING = True
logger = logging.getLogger(__name__)

class InputStreamEventAndEpochSourceNode(BaseSpikeSource, Node):
    
    _input_specs = {'in': {}}
    _output_specs = {'out': {}}

    def __init__(
            self, **kargs):
        BaseEventAndEpoch.__init__(self)
        Node.__init__(self, **kargs)
        self._t_start = 0
        self._t_stop = 0
    
    def _configure(
        self, 
        get_with_copy=False, get_with_join=True,
        return_type='spikes', buffer_size=500):
        self.get_with_copy = get_with_copy
        self.get_with_join = get_with_join
        self.return_type = return_type
        self.buffer_size = buffer_size
    
    def _initialize(self):
        self.channel_names = []
        self.channel_indexes = []
        for list_idx, chan_info in enumerate(self.input.params['channel_info']):
            if 'channel_index' in chan_info:
                chanIdx = chan_info['channel_index']
            else:
                chanIdx = list_idx
            self.channel_indexes.append(chanIdx)
            if 'name' in chan_info:
                self.channel_names.append(chan_info['name'])
            else:
                self.channel_names.append('{}_{}'.format(self.input.name, chanIdx))
        #
        self.sorted_by_time = self.input.params['sorted_by_time']
        bufferParams = {
            key: self.input.params[key]
            for key in ['double', 'axisorder', 'fill']}
        bufferParams['shmem'] = True if (self.input.params['transfermode'] == 'sharedmem') else None
        self.buffers_by_channel = {
            chanIdx: RingBuffer(
                shape=(self.buffer_size,),
                dtype=self.input.params['dtype'],
                **bufferParams)
            for chanIdx in self.channel_indexes
            }
        self.poller = ThreadPollInput(self.input, return_data=True)
        self.poller.new_data.connect(self.event_received)

    def event_received(self, ptr, data):
        # print("Event data received: %d %s" % (ptr, data.shape))
        if self.sorted_by_time:
            # sort by channel instead
            sort_indices = np.argsort(data['channel'].flatten())
            data = data[sort_indices]
        # Get the indices where shifts (IDs change) occur
        unique_chans, cut_idx = np.unique(data['channel'].flatten(), return_index=True)
        grouped = np.split(data, cut_idx)[1:]
        for chanIdx, group in zip(unique_chans, grouped):
            self.buffers_by_channel[chanIdx].new_chunk(group)
            #~print(f'{chanIdx}; {group.shape}')
        #~print('---------')
        return

    def _start(self):
        self.poller.start()

    def _stop(self):
        self.poller.stop()
        self.poller.wait()
    
    def _close(self):
        if self.running():
            self.stop()

    @property
    def nb_channel(self):
        return self.input.params['nb_channel']
        
    @property
    def t_start(self):
        return self._t_start

    @property
    def t_stop(self):
        return self._t_stop

    def get_channel_name(self, chan=0):
        return self.channel_names[chan]

    def get_size(self, chan=0):
        return

    def get_chunk(self, chan=0, i_start=None, i_stop=None):
        chanIdx = self.channel_indexes[chan]
        this_buffer = self.buffers_by_channel[chanIdx]
        sig_chunk = this_buffer.get_data(
            i_start, i_stop,
            copy=self.get_with_copy, join=self.get_with_join)
        #
        ev_times = sig_chunk['timestamp'] / 3e4
        if self.return_type == 'spikes':
            return ev_times
        ev_labels = sig_chunk['class_id']
        if self.return_type == 'events':
            return ev_times, ev_labels
        ev_durations = np.ones(ev_times.shape) * 1e-3
        if self.return_type == 'epochs':
            return ev_times, ev_durations, ev_labels

    def get_chunk_by_time(
            self, chan=0,
            t_start=None, t_stop=None):
        chanIdx = self.channel_indexes[chan]
        this_buffer = self.buffers_by_channel[chanIdx]
        sig_chunk = this_buffer.get_data(
            this_buffer.first_index(), this_buffer.index(),
            copy=self.get_with_copy, join=self.get_with_join)
        all_times = sig_chunk['timestamp'] / 3e4
        # print(all_times)
        timeMask = (all_times >= t_start) & (all_times < t_stop)
        ev_times = all_times[timeMask]
        # i1 = np.searchsorted(all_times, t_start, side='left')
        # i2 = np.searchsorted(all_times, t_stop, side='left')
        # sl = slice(i1, i2+1)
        # ev_times = all_times[sl]
        if self.return_type == 'spikes':
            return ev_times
        ev_labels = this_buffer.buffer['class_id'][timeMask]
        if self.return_type == 'events':
            return ev_times, ev_labels
        ev_durations = np.ones(ev_times.shape) * 1e-3
        if self.return_type == 'epochs':
            return ev_times, ev_durations, ev_labels


class InputStreamAnalogSignalSource(BaseAnalogSignalSource, Node):

    _input_specs = {'in': {}}
    _output_specs = {'out': {}}

    def __init__(self, **kargs):
        BaseAnalogSignalSource.__init__(self)
        Node.__init__(self, **kargs)
        self._t_start = 0
        self._t_stop = 0

    def _configure(
        self,
        get_with_copy=False, get_with_join=True):
        #
        self.get_with_copy = get_with_copy
        self.get_with_join = get_with_join
        #
    def _initialize(self):
        self.has_custom_dtype = self.input.params['dtype'].names is not None
        self.sample_rate = float(self.input.params['sample_rate'])
        self.reference_signal = None
        # set_buffer(self, size=None, double=True, axisorder=None, shmem=None, fill=None)
        bufferParams = {
            key: self.input.params[key] for key in ['double', 'axisorder', 'fill']}
        bufferParams['size'] = self.input.params['buffer_size']
        if (self.input.params['transfermode'] == 'sharedmem'):
            if 'shm_id' in self.input.params:
                bufferParams['shmem'] = self.input.params['shm_id']
            else:
                bufferParams['shmem'] = True
        else:
            bufferParams['shmem'] = None
        self.input.set_buffer(**bufferParams)
        # There are many ways to poll for data from the input stream. In this
        # case, we will use a background thread to monitor the stream and emit
        # a Qt signal whenever data is available.
        self.poller = ThreadPollInput(self.input)
        # self.pollers[inputname].new_data.connect(self.analogsignal_received)
        #
        self._t_start = 0.
        self._t_stop = self.input.buffer.shape[0] / self.sample_rate + self._t_start
        #
        self.channel_names = []
        self.channel_indexes = []
        for list_idx, chan_info in enumerate(self.input.params['channel_info']):
            if 'channel_index' in chan_info:
                chanIdx = chan_info['channel_index']
            else:
                chanIdx = list_idx
            self.channel_indexes.append(chanIdx)
            if 'name' in chan_info:
                self.channel_names.append(chan_info['name'])
            else:
                self.channel_names.append('{}_{}'.format(self.input.name, chanIdx))
        self.signals = self.input.buffer
        return

    @property
    def nb_channel(self):
        return self.input.buffer.shape[1]

    def get_channel_name(self, chan=0):
        return self.channel_names[chan]

    @property
    def t_start(self):
        return self._t_start

    @property
    def t_stop(self):
        return self._t_stop

    def get_length(self):
        return self.input.buffer.shape[0]

    def get_chunk(self, i_start=None, i_stop=None):
        # print('InputStreamAnalogSignalSource; t_start = {:.3f}'.format(self._t_start))
        '''
        start, stop = (
            self.signals._interpret_index(i_start),
            self.signals._interpret_index(i_stop))
            '''
        sig_chunk = self.input.get_data(
            i_start, i_stop,
            copy=self.get_with_copy, join=self.get_with_join)
        if self.has_custom_dtype:
            sig_chunk = sig_chunk['value']
        if self.reference_signal is not None:
            sig_chunk = sig_chunk - sig_chunk[:, self.reference_signal][:, None]
        return sig_chunk
   

class InputStreamEventAndEpochSource(BaseSpikeSource, QT.QObject):
    
    def __init__(
            self, stream,
            get_with_copy=False, get_with_join=True,
            return_type='spikes', buffer_size=500,
            parent=None
            ):
        QT.QObject.__init__(self, parent)
        BaseEventAndEpoch.__init__(self)
        self.input = stream
        self._t_start = 0
        self._t_stop = 0
        self.buffer_size = buffer_size
        #
        self.get_with_copy = get_with_copy
        self.get_with_join = get_with_join
        self.return_type = return_type
        #
        self.channel_names = []
        self.channel_indexes = []
        for list_idx, chan_info in enumerate(self.input.params['channel_info']):
            if 'channel_index' in chan_info:
                chanIdx = chan_info['channel_index']
            else:
                chanIdx = list_idx
            self.channel_indexes.append(chanIdx)
            if 'name' in chan_info:
                self.channel_names.append(chan_info['name'])
            else:
                self.channel_names.append('{}_{}'.format(self.input.name, chanIdx))
        #
        self.sorted_by_time = self.input.params['sorted_by_time']
        bufferParams = {
            key: self.input.params[key]
            for key in ['double', 'axisorder', 'fill']}
        bufferParams['shmem'] = True if (self.input.params['transfermode'] == 'sharedmem') else None
        self.buffers_by_channel = {
            chanIdx: RingBuffer(
                shape=(self.buffer_size,),
                dtype=self.input.params['dtype'],
                **bufferParams)
            for chanIdx in self.channel_indexes
            }

    def event_received(self, ptr, data):
        # print("Event data received: %d %s" % (ptr, data.shape))
        if self.sorted_by_time:
            # sort by channel instead
            sort_indices = np.argsort(data['channel'].flatten())
            data = data[sort_indices]
        # Get the indices where shifts (IDs change) occur
        unique_chans, cut_idx = np.unique(data['channel'].flatten(), return_index=True)
        grouped = np.split(data, cut_idx)[1:]
        for chanIdx, group in zip(unique_chans, grouped):
            self.buffers_by_channel[chanIdx].new_chunk(group)
            #~print(f'{chanIdx}; {group.shape}')
        #~print('---------')
        return

    @property
    def nb_channel(self):
        return self.input.params['nb_channel']
        
    @property
    def t_start(self):
        return self._t_start

    @property
    def t_stop(self):
        return self._t_stop

    def get_channel_name(self, chan=0):
        return self.channel_names[chan]

    def get_size(self, chan=0):
        return

    def get_chunk(self, chan=0, i_start=None, i_stop=None):
        chanIdx = self.channel_indexes[chan]
        this_buffer = self.buffers_by_channel[chanIdx]
        sig_chunk = this_buffer.get_data(
            i_start, i_stop,
            copy=self.get_with_copy, join=self.get_with_join)
        #
        ev_times = sig_chunk['timestamp'] / 3e4
        if self.return_type == 'spikes':
            return ev_times
        ev_labels = sig_chunk['class_id']
        if self.return_type == 'events':
            return ev_times, ev_labels
        ev_durations = np.ones(ev_times.shape) * 1e-3
        if self.return_type == 'epochs':
            return ev_times, ev_durations, ev_labels

    def get_chunk_by_time(
            self, chan=0,
            t_start=None, t_stop=None):
        chanIdx = self.channel_indexes[chan]
        this_buffer = self.buffers_by_channel[chanIdx]
        # print(f'this_buffer.buffer.shape = {this_buffer.buffer.shape}')
        sig_chunk = this_buffer.get_data(
            this_buffer.first_index(), this_buffer.index(),
            copy=self.get_with_copy, join=self.get_with_join)
        all_times = sig_chunk['timestamp'] / 3e4
        # print(all_times)
        timeMask = (all_times >= t_start) & (all_times < t_stop)
        ev_times = all_times[timeMask]
        # i1 = np.searchsorted(all_times, t_start, side='left')
        # i2 = np.searchsorted(all_times, t_stop, side='left')
        # sl = slice(i1, i2+1)
        # ev_times = all_times[sl]
        if self.return_type == 'spikes':
            return ev_times
        ev_labels = this_buffer.buffer['class_id'][timeMask]
        if self.return_type == 'events':
            return ev_times, ev_labels
        ev_durations = np.ones(ev_times.shape) * 1e-3
        if self.return_type == 'epochs':
            return ev_times, ev_durations, ev_labels

 
class InputStreamAnalogSignalSource(BaseAnalogSignalSource, QT.QObject):
    def __init__(
        self, stream, parent=None,
        get_with_copy=False, get_with_join=True):
        QT.QObject.__init__(self, parent)
        BaseAnalogSignalSource.__init__(self)
        #
        self.input = stream
        self.get_with_copy = get_with_copy
        self.get_with_join = get_with_join
        #
        self.has_custom_dtype = self.input.params['dtype'].names is not None
        self.signals = self.input.buffer
        self.sample_rate = float(self.input.params['sample_rate'])
        self.reference_signal = None
        #
        self._t_start = 0.
        #~
        self._t_stop = self.input.buffer.shape[0] / self.sample_rate + self._t_start
        #
        self.channel_names = []
        self.channel_indexes = []
        for list_idx, chan_info in enumerate(self.input.params['channel_info']):
            if 'channel_index' in chan_info:
                chanIdx = chan_info['channel_index']
            else:
                chanIdx = list_idx
            self.channel_indexes.append(chanIdx)
            if 'name' in chan_info:
                self.channel_names.append(chan_info['name'])
            else:
                self.channel_names.append('{}_{}'.format(self.input.name, chanIdx))
        return

    @property
    def nb_channel(self):
        return self.input.buffer.shape[1]

    def get_channel_name(self, chan=0):
        return self.channel_names[chan]

    @property
    def t_start(self):
        return self._t_start

    @property
    def t_stop(self):
        return self._t_stop

    def get_length(self):
        return self.input.buffer.shape[0]

    def get_chunk(self, i_start=None, i_stop=None):
        # if LOGGING:
        #     logger.info(f"{self.input.name: >10}-buf[{id(self.input.buffer):X}].get_data(dsize={i_stop - i_start})")
        sig_chunk = self.input.get_data(
            i_start, i_stop,
            copy=self.get_with_copy, join=self.get_with_join)
        if self.has_custom_dtype:
            sig_chunk = sig_chunk['value']
        if self.reference_signal is not None:
            sig_chunk = sig_chunk - sig_chunk[:, self.reference_signal][:, None]
        return sig_chunk
    

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


defaultNavParams = dict(
    parent=None, show_play=True, show_step=True,
    is_playing=True,
    show_scroll_time=True, show_spinbox=True,
    show_label_datetime=False, datetime0=None,
    datetime_format='%Y-%m-%d %H:%M:%S',
    show_global_xsize=True, show_auto_scale=True,
    )

class NodeNavigationToolbar(NavigationToolBar):
    def __init__(
            self, parent=None, show_play=True, show_step=True,
            is_playing=True,
            show_scroll_time=True, show_spinbox=True,
            show_label_datetime=False, datetime0=None,
            datetime_format='%Y-%m-%d %H:%M:%S',
            show_global_xsize=True, show_auto_scale=True,
            ):
        QT.QWidget.__init__(self, parent)

        if QT_LIB == 'PyQt6':
            self.setSizePolicy(
                QT.QSizePolicy.Policy.Minimum, QT.QSizePolicy.Policy.Maximum)
        else:
            self.setSizePolicy(
                QT.QSizePolicy.Minimum, QT.QSizePolicy.Maximum)

        self.mainlayout = QT.QVBoxLayout()
        self.setLayout(self.mainlayout)

        self.is_playing = is_playing
        self.show_play = show_play
        self.show_step = show_step
        self.show_scroll_time = show_scroll_time
        self.show_spinbox = show_spinbox
        self.show_label_datetime = show_label_datetime
        self.show_global_xsize = show_global_xsize

        self.datetime0 = datetime0
        self.datetime_format = datetime_format

        if show_scroll_time:
            #~ self.slider = QSlider()
            self.scroll_time = QT.QScrollBar(
                orientation=orientation_to_qt['horizontal'],
                minimum=0, maximum=1000)
            self.mainlayout.addWidget(self.scroll_time)
            if not self.get_playing():
                # do not connect this if it will play by default
                self.scroll_time.valueChanged.connect(self.on_scroll_time_changed)

            #TODO min/max/step
            #~ self.scroll_time.valueChanged.disconnect(self.on_scroll_time_changed)
            #~ self.scroll_time.setValue(int(sr*t))
            #~ self.scroll_time.setPageStep(int(sr*self.xsize))
            #~ self.scroll_time.valueChanged.connect(self.on_scroll_time_changed)
            #~ self.scroll_time.setMinimum(0)
            #~ self.scroll_time.setMaximum(length)

        h = QT.QHBoxLayout()
        h.addStretch()
        self.mainlayout.addLayout(h)

        if show_play:
            but = QT.QPushButton(icon=QT.QIcon(':/media-playback-start.svg'))
            but.clicked.connect(self.on_play)
            h.addWidget(but)

            but = QT.QPushButton(icon=QT.QIcon(':/media-playback-stop.svg'))
            #~ but = QT.QPushButton(QT.QIcon(':/media-playback-stop.png'), '')
            but.clicked.connect(self.on_stop_pause)
            h.addWidget(but)

            #trick for separator
            h.addWidget(QT.QFrame(
                frameShape=QT.QFrame.VLine,
                frameShadow=QT.QFrame.Sunken))

            # add spacebar shortcut for play/pause
            play_pause_shortcut = QT.QShortcut(self)
            play_pause_shortcut.setKey(QT.QKeySequence(' '))
            play_pause_shortcut.activated.connect(self.on_play_pause_shortcut)

        self.steps = ['60 s', '10 s', '1 s', '100 ms', '50 ms', '5 ms', '1 ms', '200 us']

        if show_step:
            but = QT.QPushButton('<')
            but.clicked.connect(self.prev_step)
            h.addWidget(but)

            self.combo_step = QT.QComboBox()
            self.combo_step.addItems(self.steps)
            self.combo_step.setCurrentIndex(2)
            h.addWidget(self.combo_step)

            self.on_change_step(None)
            self.combo_step.currentIndexChanged.connect(self.on_change_step)

            but = QT.QPushButton('>')
            but.clicked.connect(self.next_step)
            h.addWidget(but)

            self.speedSpinLabel = QT.QLabel('Refresh rate (Hz):')
            h.addWidget(self.speedSpinLabel)
            self.speedSpin = pg.SpinBox(bounds=(1, 50.), step=5, value=10.)
            if 'compactHeight' in self.speedSpin.opts:  # pyqtgraph >= 0.11.0
                self.speedSpin.setOpts(compactHeight=False)
            h.addWidget(self.speedSpin)

            #trick for separator
            h.addWidget(QT.QFrame(frameShape=QT.QFrame.VLine, frameShadow=QT.QFrame.Sunken))

            # add shortcuts for stepping through time and changing step size
            shortcuts = [
                {'key': QT.Qt.Key_Left,  'callback': self.prev_step},
                {'key': QT.Qt.Key_Right, 'callback': self.next_step},
                {'key': QT.Qt.Key_Up,    'callback': self.increase_step},
                {'key': QT.Qt.Key_Down,  'callback': self.decrease_step},
                {'key': 'a',             'callback': self.prev_step},
                {'key': 'd',             'callback': self.next_step},
                {'key': 'w',             'callback': self.increase_step},
                {'key': 's',             'callback': self.decrease_step},
            ]
            for s in shortcuts:
                shortcut = QT.QShortcut(self)
                shortcut.setKey(QT.QKeySequence(s['key']))
                shortcut.activated.connect(s['callback'])

        self.time_label = QT.QLabel('Time: 0.000 sec')
        h.addWidget(self.time_label)
        h.addWidget(QT.QFrame(frameShape=QT.QFrame.VLine, frameShadow=QT.QFrame.Sunken))
        if show_spinbox:
            h.addWidget(QT.QLabel('Seek to (sec):'))
            self.spinbox_time =pg.SpinBox(decimals = 8, bounds = (-np.inf, np.inf),step = 0.05, siPrefix=False, suffix='', int=False)
            if 'compactHeight' in self.spinbox_time.opts:  # pyqtgraph >= 0.11.0
                self.spinbox_time.setOpts(compactHeight=False)
            h.addWidget(self.spinbox_time)
            #trick for separator
            h.addWidget(QT.QFrame(frameShape=QT.QFrame.VLine, frameShadow=QT.QFrame.Sunken))
            # do not connect this if it will play by default
            if not self.get_playing():
                self.spinbox_time.valueChanged.connect(self.on_spinbox_time_changed)

        if show_label_datetime:
            assert self.datetime0 is not None
            self.label_datetime = QT.QLabel('')
            h.addWidget(self.label_datetime)
            #trick for separator
            h.addWidget(QT.QFrame(frameShape=QT.QFrame.VLine, frameShadow=QT.QFrame.Sunken))

        if show_global_xsize:
            h.addWidget(QT.QLabel('Time width (s):'))
            self.spinbox_xsize =pg.SpinBox(value=3., decimals = 8, bounds = (0.001, np.inf),step = 0.1, siPrefix=False, suffix='', int=False)
            if 'compactHeight' in self.spinbox_xsize.opts:  # pyqtgraph >= 0.11.0
                self.spinbox_xsize.setOpts(compactHeight=False)
            h.addWidget(self.spinbox_xsize)
            #~ self.spinbox_xsize.valueChanged.connect(self.on_spinbox_xsize_changed)
            self.spinbox_xsize.valueChanged.connect(self.xsize_changed.emit)
            #trick for separator
            h.addWidget(QT.QFrame(frameShape=QT.QFrame.VLine, frameShadow=QT.QFrame.Sunken))

        if show_auto_scale:
            but = QT.PushButton('Auto scale')
            h.addWidget(but)
            but.clicked.connect(self.auto_scale_requested.emit)
            #~ h.addWidget(QT.QFrame(frameShape=QT.QFrame.VLine, frameShadow=QT.QFrame.Sunken))
        h.addStretch()

        # all in s
        self.t = 0 #  s
        self.set_start_stop(0., 0.1)

    def on_play(self):
        # if play is currently disabled, we are starting it;
        # disconnect these
        if not self.get_playing():
            self.spinbox_time.valueChanged.disconnect(self.on_spinbox_time_changed)
            self.scroll_time.valueChanged.disconnect(self.on_scroll_time_changed)
            self.set_playing(True)
            self.play_pause_signal.emit(True)

    def on_stop_pause(self):
        # if play is currently enabled, we are stopping it;
        # connect these
        if self.get_playing():
            self.spinbox_time.valueChanged.connect(self.on_spinbox_time_changed)
            self.scroll_time.valueChanged.connect(self.on_scroll_time_changed)
            self.set_playing(False)
            self.play_pause_signal.emit(False)

    def on_play_pause_shortcut(self):
        if self.get_playing():
            self.on_stop_pause()
        else:
            self.on_play()

    def seek(
            self, t, refresh_scroll=True,
            refresh_spinbox=True, emit=True):
        self.t = t
        if (self.t < self.t_start):
            self.t = self.t_start
        if (self.t > self.t_stop):
            self.t = self.t_stop

        self.time_label.setText(f'Time: {t:.3f} sec')

        if refresh_scroll and self.show_scroll_time:
            if not self.get_playing():
                self.scroll_time.blockSignals(True)
            pos = int((self.t - self.t_start)/(self.t_stop - self.t_start)*1000.)
            self.scroll_time.setValue(pos)
            if not self.get_playing():
                self.scroll_time.blockSignals(False)
            
        if refresh_spinbox and self.show_spinbox:
            if not self.get_playing():
                self.spinbox_time.blockSignals(True)
            self.spinbox_time.setValue(t)
            if not self.get_playing():
                self.spinbox_time.blockSignals(False)

        if self.show_label_datetime:
            dt = self.datetime0 + datetime.timedelta(seconds=self.t)
            self.label_datetime.setText(dt.strftime(self.datetime_format))

        if emit:
            self.time_changed.emit(self.t)

    def on_xsize_changed(
            self, xsize):
        self.xsize = xsize
        #~ print('on_xsize_changed', xsize)
        return

    def auto_scale(self):
        #~ print('on_xsize_changed', xsize)
        '''
        self.timer.blockSignals(True)
        time.sleep(500e-3)
        self.timer.blockSignals(False)
        '''
        return


class NodeMainViewer(MainViewer):
    seek_time = QT.pyqtSignal(float)

    def __init__(
        self, node=None, time_reference_source=None,
        speed=None,
        debug=False, settings_name=None, parent=None,
        global_xsize_zoom=False, navigation_params={}):
        self.node = node
        #
        self.time_reference_source = None
        self.source_sample_rate = None
        self.source_buffer_dur = None
        self.t_head = None
        if time_reference_source is not None:
            self.set_time_reference_source(time_reference_source)
        #
        if speed is not None:
            self.speed = speed
        else:
            self.speed = 1.
            
        QT.QMainWindow.__init__(self, parent)
        #TODO settings
        #http://www.programcreek.com/python/example/86789/PyQt5.QtCore.QSettings
        self.debug = debug
        self.settings_name = settings_name
        if self.settings_name is not None:
            pyver = '.'.join(str(e) for e in sys.version_info[0:3])
            appname = 'ephyviewer'+'_py'+pyver
            self.settings = QT.QSettings(appname, self.settings_name)
        self.global_xsize_zoom = global_xsize_zoom
        self.xsize = 3.
        self.xratio = 0.3
        self.setDockNestingEnabled(True)
        
        self.threads = []
        self.viewers = OrderedDict()
        
        navParams = defaultNavParams.copy()
        navParams.update(navigation_params)
        #~print(navParams)
        self.navigation_toolbar = NodeNavigationToolbar(**navParams)

        dock = self.navigation_dock = QT.QDockWidget('navigation', self)
        dock.setObjectName('navigation')
        dock.setWidget(self.navigation_toolbar)
        dock.setTitleBarWidget(QT.QWidget())  # hide the widget title bar
        dock.setFeatures(QT.DockWidget.NoDockWidgetFeatures)  # prevent accidental movement and undockingx
        self.addDockWidget(QT.TopDockWidgetArea, dock)

        self.navigation_toolbar.speedSpin.setValue(self.speed)
        self.timer = RefreshTimer(interval=self.speed ** -1, node=self)
        self.timer.timeout.connect(self.refresh)
        # self.timer.start()

        self.t_refresh = 0.
        self.last_t_refresh = -1.
        self.refresh_enabled = True

        self.navigation_toolbar.time_changed.connect(self.on_time_changed)
        self.navigation_toolbar.xsize_changed.connect(self.on_xsize_changed)
        # self.navigation_toolbar.auto_scale_requested.connect(self.auto_scale)
        self.navigation_toolbar.speedSpin.valueChanged.connect(self.on_change_speed)
        self.navigation_toolbar.play_pause_signal.connect(self.set_refresh_enable)
        self.load_one_setting('navigation_toolbar', self.navigation_toolbar)
        # self.showMaximized()

    def set_refresh_enable(self, value):
        self.refresh_enabled = value
        # print(f'mainviewer.refresh_enabled = {value}')

    def update_t_head(self, ptr, data):
        self.t_head = ptr / self.source_sample_rate
        return

    def set_time_reference_source(self, source):
        # pdb.set_trace()
        stream_name = source.input.name
        self.source_sample_rate = source.sample_rate
        self.source_buffer_dur = source.signals.shape[0] / source.sample_rate
        poller = self.node.pollers[stream_name]
        poller.new_data.connect(self.update_t_head)
        self.time_reference_source = source

    def add_view(
        self, widget, connect_seek_time=True,
        **kwargs):
        MainViewer.add_view(self, widget, **kwargs)
        if connect_seek_time:
            widget.connect_to_seek(self.seek_time)
        return
        
    def reset_navbar_bounds(self, t_max):
        nav_t_start = min(
            self.navigation_toolbar.t_start,
            t_max - self.source_buffer_dur)
        nav_t_stop = max(
            self.navigation_toolbar.t_stop,
            t_max)
        self.navigation_toolbar.set_start_stop(
            nav_t_start, nav_t_stop, seek=False)

    def refresh(self):
        if self.t_head is not None:
            t_max = self.t_head
            if self.refresh_enabled:
                if self.navigation_toolbar.get_playing():
                    self.t_refresh = t_max - self.xsize * (1 - self.xratio)
                else:
                    self.t_refresh = self.navigation_toolbar.t
                if self.t_refresh != self.last_t_refresh:
                    self.seek_time.emit(self.t_refresh)
                    #
                    if self.navigation_toolbar.get_playing():
                        self.navigation_toolbar.seek(
                            self.t_refresh, refresh_spinbox=False, emit=False)
            self.reset_navbar_bounds(t_max)
            self.last_t_refresh = self.t_refresh

    def on_change_speed(self, speed):
        self.speed = speed
        self.timer.set_interval(speed ** -1)

    def start_viewers(self):
        for _, d in self.viewers.items():
            if hasattr(d['widget'], 'start_threads'):
                d['widget'].start_threads()
        self.timer.start()

    def closeEvent(self, event):
        self.timer.stop()
        for name, viewer in self.viewers.items():
            viewer['widget'].close()
        for i, thread in enumerate(self.threads):
            thread.quit()
            thread.wait()
        self.save_all_settings()
        event.accept()


class RefreshTimer(QT.QThread):
    timeout = QT.pyqtSignal()
    
    def __init__(
            self,
            interval=100e-3, node=None,
            verbose=False, parent=None):
        QT.QThread.__init__(self, parent)
        self.verbose = verbose
        self.interval = interval
        self.node = weakref.ref(node)
        self.setObjectName(f'RefreshTimer_')
        #
        self.mutex = QT.QMutex()
        self.lock = QT.QMutexLocker(self.mutex)
        self.running = False
        self.is_disabled = False
        atexit.register(self.stop)
        #
    def run(self):
        with self.lock:
            self.running = True
            interval = self.interval
            is_disabled = self.is_disabled
        next_time = time.perf_counter() + interval
        while True:
            # print('RefreshTimer: sleeping for {:.3f} sec'.format(max(0, next_time - time.perf_counter())))
            time.sleep(max(0, next_time - time.perf_counter()))
            if not is_disabled:
                self.timeout.emit()
            # print('t={:.3f} RefreshViewer timeout()'.format(time.perf_counter()))
            # check the interval, in case it has changed
            with self.lock:
                interval = self.interval
                running = self.running
                is_disabled = self.is_disabled
            if not running:
                break
            # skip tasks if we are behind schedule:
            next_time += (time.perf_counter() - next_time) // interval * interval + interval
        return

    def set_interval(self, interval):
        with self.lock:
            self.interval = interval

    def disable(self):
        with self.lock:
            self.is_disabled = True

    def enable(self):
        with self.lock:
            self.is_disabled = False

    def stop(self):
        with self.lock:
            self.running = False


class RippleTriggerAccumulator(TriggerAccumulator):
    """
    Here the list of theses attributes with shape and dtype. **N** is the total 
    number of peak detected. **M** is the number of selected peak for
    waveform/feature/cluser. **C** is the number of clusters
      * all_peaks (N, ) dtype = {0}
      * clusters (c, ) dtype= {1}
      * some_peaks_index (M) int64
      * centroids_median (C, width, nb_channel) float32
      * centroids_mad (C, width, nb_channel) float32
      * centroids_mean (C, width, nb_channel) float32
      * centroids_std (C, width, nb_channel) float32
    """.format(_dtype_peak, _dtype_cluster)
    
    _input_specs = {
        'signals' : dict(streamtype = 'signals'), 
        'events' : dict(streamtype = 'events',  shape = (-1,)), #dtype ='int64',
        }
    _output_specs = {}
    
    _default_params = [
        {'name': 'left_sweep', 'type': 'float', 'value': -.1, 'step': 0.1,'suffix': 's', 'siPrefix': True},
        {'name': 'right_sweep', 'type': 'float', 'value': .2, 'step': 0.1, 'suffix': 's', 'siPrefix': True},
        { 'name' : 'stack_size', 'type' :'int', 'value' : 1000,  'limits':[1,np.inf] },
            ]
    
    new_chunk = QT.pyqtSignal(int)
    
    def __init__(
            self, parent=None, **kargs,
            ):
        TriggerAccumulator.__init__(self, parent=parent, **kargs)
    
    def _configure(
            self, max_stack_size=2000, max_xsize=2.,
            channel_group=None):
        """
        Arguments
        ---------------
        max_stack_size: int
            maximum size for the event size
        max_xsize: int 
            maximum sample chunk size
        events_dtype_field : None or str
            Standart dtype for 'events' input is 'int64',
            In case of complex dtype (ex : dtype = [('index', 'int64'), ('label', 'S12), ) ] you can precise which
            filed is the index.
        """
        self.params.sigTreeStateChanged.connect(
            self.on_params_change)
        self.max_stack_size = max_stack_size
        self.events_dtype_field = 'timestamp'
        self.params.param('stack_size').setLimits([1, self.max_stack_size])
        self.max_xsize = max_xsize
        self.channel_group = channel_group
        self.channel_groups = {0: channel_group}

    def after_input_connect(self, inputname):
        if inputname == 'signals':
            self.nb_channel = self.inputs['signals'].params['shape'][1]
            self.sample_rate = self.inputs['signals'].params['sample_rate']
            self.nip_sample_period = self.inputs['signals'].params['nip_sample_period']
        elif inputname == 'events':
            dt = np.dtype(self.inputs['events'].params['dtype'])
            assert self.events_dtype_field in dt.names, 'events_dtype_field not in input dtype {}'.format(dt)

    def _initialize(self):
        # set_buffer(self, size=None, double=True, axisorder=None, shmem=None, fill=None)
        bufferParams = {
            key: self.inputs['signals'].params[key] for key in ['double', 'axisorder', 'fill']}
        bufferParams['size'] = self.inputs['signals'].params['buffer_size']
        # print(f"self.inputs['signals'].params['buffer_size'] = {self.inputs['signals'].params['buffer_size']}")
        if (self.inputs['signals'].params['transfermode'] == 'sharedmem'):
            if 'shm_id' in self.inputs['signals'].params:
                bufferParams['shmem'] = self.inputs['signals'].params['shm_id']
            else:
                bufferParams['shmem'] = True
        else:
            bufferParams['shmem'] = None
        self.inputs['signals'].set_buffer(**bufferParams)
        #
        self.trig_poller = ThreadPollInput(self.inputs['events'], return_data=True)
        self.trig_poller.new_data.connect(self.on_new_trig)
        
        self.limit_poller = ThreadPollInputUntilPosLimit(self.inputs['signals'])
        self.limit_poller.limit_reached.connect(self.on_limit_reached)
        
        self.stack_lock =  Mutex()
        self.wait_thread_list = []
        self.recreate_stack()
        
        self.nb_segment = 1
        self.total_channel = self.nb_channel
        self.source_dtype = np.dtype('float64')

        clean_shape = lambda shape: tuple(int(e) for e in shape)
        self.segment_shapes = [
            clean_shape(self.get_segment_shape(s))
            for s in range(self.nb_segment)]

        channel_info = self.inputs['signals'].params['channel_info']
        self.all_channel_names = [
            item['name'] for item in channel_info
            ]
        self.datasource = DummyDataSource(self.all_channel_names)
        stim_channels = [
            item['channel_index']
            for item in self.inputs['events'].params['channel_info']]
        self.clusters = np.zeros(shape=(len(stim_channels),), dtype=_dtype_cluster)
        self.clusters['cluster_label'] = stim_channels
        self._all_peaks_buffer = RingBuffer(
            shape=(self.params['stack_size'], 1), dtype=_dtype_peak,
            fill=_dtype_peak_zero)
        self.some_peaks_index = np.arange(self._all_peaks_buffer.shape[0])
        self.n_spike_for_centroid = 500

        n_left = self.limit1
        n_right = self.limit2
        #
        self.centroids_median = np.zeros((self.cluster_labels.size, n_right - n_left, self.nb_channel), dtype=self.source_dtype)
        self.centroids_mad = np.zeros((self.cluster_labels.size, n_right - n_left, self.nb_channel), dtype=self.source_dtype)
        self.centroids_mean = np.zeros((self.cluster_labels.size, n_right - n_left, self.nb_channel), dtype=self.source_dtype)
        self.centroids_std = np.zeros((self.cluster_labels.size, n_right - n_left, self.nb_channel), dtype=self.source_dtype)

    def on_new_trig(self, trig_num, trig_indexes):
        # print(f'on_new_trig {trig_indexes}')
        # if LOGGING:
        #     logger.info(f'on_new_trig: {trig_indexes}')
        # add to all_peaks
        adj_index = (
            trig_indexes[self.events_dtype_field].flatten() / self.nip_sample_period).astype('int64')
        for trig_index in adj_index:
            self.limit_poller.append_limit(trig_index + self.limit2)
        data = np.zeros(trig_indexes.shape, dtype=_dtype_peak)
        data['timestamp'] = trig_indexes[self.events_dtype_field].flatten()
        data['index'] = adj_index
        data['cluster_label'] = trig_indexes['channel'].flatten().astype('int64')
        data['channel'] = 0
        data['segment'] = 0
        self._all_peaks_buffer.new_chunk(data[:, None])
        # print(f'self._all_peaks_buffer.new_chunk(); self._all_peaks_buffer.index() = {self._all_peaks_buffer.index()}')
                    
    def on_limit_reached(self, limit_index):
        # if LOGGING:
        #     logger.info(f'on limit reached: {limit_index-self.size}:{limit_index}')
        # arr = self.inputs['signals'].get_data(limit_index-self.size, limit_index)
        arr = self.get_signals_chunk(i_start=limit_index-self.size, i_stop=limit_index)
        if arr is not None:
            # with self.stack_lock:
            #     self.stack[self.stack_pos,:,:] = arr['value']
            self.stack.new_chunk(arr.reshape(1, (self.limit2 - self.limit1) * self.nb_channel))
            # self.new_chunk.emit(self.stack.index())
            #
            # print(f"on_limit_reached,\nself.stack[self.stack_pos,:,:] = {self.stack[self.stack_pos,:,:]}\nself.total_trig = {self.total_trig}")
            # self.stack_pos += 1
            # self.stack_pos = self.stack_pos % self.params['stack_size']
            # self.total_trig += 1
            # print(f'self.stack.new_chunk(); self.total_trig = {self.total_trig}')
            # self.new_chunk.emit(self.total_trig)

    def recreate_stack(self):
        self.limit1 = l1 = int(self.params['left_sweep'] * self.sample_rate)
        self.limit2 = l2 = int(self.params['right_sweep'] * self.sample_rate)
        self.size = l2 - l1
        
        self.t_vect = np.arange(l2-l1)/self.sample_rate + self.params['left_sweep']
        '''
        with self.stack_lock:
            self.stack = np.zeros(
                (self.params['stack_size'], l2-l1, self.nb_channel),
                dtype = 'float64')'''
        self.stack = RingBuffer(
            shape=(self.params['stack_size'], (l2-l1) * self.nb_channel),
            dtype='float64')
        # self.stack_pos = 0
        # self.total_trig = 0
        self.limit_poller.reset()

    def get_geometry(self):
        """
        Get the geometry for a given channel group in a numpy array way.
        """
        geometry = [ self.channel_group['geometry'][chan] for chan in self.channel_group['channels'] ]
        geometry = np.array(geometry, dtype='float64')
        return geometry
    
    def get_channel_distances(self):
        geometry = self.get_geometry()
        distances = sklearn.metrics.pairwise.euclidean_distances(geometry)
        return distances
    
    def get_channel_adjacency(self, adjacency_radius_um=None):
        assert adjacency_radius_um is not None
        channel_distances = self.get_channel_distances()
        channels_adjacency = {}
        for c in range(self.nb_channel):
            nearest, = np.nonzero(channel_distances[c, :] < adjacency_radius_um)
            channels_adjacency[c] = nearest
        return channels_adjacency

    def get_segment_length(self, seg_num):
        """
        Segment length (in sample) for a given segment index
        """
        return self.inputs['signals'].buffer.index()
    
    def get_segment_shape(self, seg_num):
        return self.inputs['signals'].buffer.shape

    def get_signals_chunk(
        self, seg_num=0, chan_grp=0,
        signal_type='initial',
        i_start=None, i_stop=None, pad_width=0):
        """
        Get a chunk of signal for for a given segment index and channel group.
        
        Parameters
        ------------------
        seg_num: int
            segment index
        chan_grp: int
            channel group key
        i_start: int or None
           start index (included)
        i_stop: int or None
            stop index (not included)
        pad_width: int (0 default)
            Add optional pad on each sides
            usefull for filtering border effect
        
        """
        channels = self.channel_group['channels']
        #
        sig_chunk_size = i_stop - i_start
        first = self.inputs['signals'].buffer.first_index()
        last = self.inputs['signals'].buffer.index()
        #
        after_padding = False
        if i_start >= last or i_stop <= first:
            return np.zeros((sig_chunk_size, len(channels)), dtype='float64')
        if i_start < first:
            pad_left = first - i_start
            i_start = first
            after_padding = True
        else:
            pad_left = 0
        if i_stop > last:
            pad_right = i_stop - last
            i_stop = last
            after_padding = True
        else:
            pad_right = 0
        #
        data = self.inputs['signals'].get_data(i_start, i_stop, copy=False, join=True)
        data = data['value']
        data = data[:, channels]
        #
        if after_padding:
            # finalize padding on border
            data2 = np.zeros((data.shape[0] + pad_left + pad_right, data.shape[1]), dtype=data.dtype)
            data2[pad_left:data2.shape[0]-pad_right, :] = data
            return data2
        return data

    def get_some_waveforms(
        self, seg_num=None, chan_grp=0,
        peak_sample_indexes=None, peaks_index=None,
        n_left=None, n_right=None, waveforms=None, channel_indexes=None):
        """
        Exctract some waveforms given sample_indexes
        seg_num is int then all spikes come from same segment
        if seg_num is None then seg_nums is an array that contain seg_num for each spike.
        """
        if channel_indexes is None:
            channel_indexes = slice(None)
        #
        if peaks_index is None:
            # print(f'get_some_waveforms( peak_sample_indexes = {peak_sample_indexes}')
            assert peak_sample_indexes is not None, 'Provide sample_indexes'
            peaks_index = np.flatnonzero(np.isin(self.all_peaks['index'], peak_sample_indexes))
        # import pdb; pdb.set_trace()
        # peaks_index = self.params['stack_size'] - peaks_index - 1
        # print(f'get_some_waveforms( peaks_index = {peaks_index}')
        '''
        with self.stack_lock:
            waveforms = self.stack[:, :, channel_indexes]
            waveforms = waveforms[peaks_index, :, :]
            '''
        first = self.stack.first_index()
        if isinstance(peaks_index, (int, np.int64)):
            waveforms_flat = self.stack.get_data(start=peaks_index+first, stop=peaks_index+first+1).copy()
            waveforms = waveforms_flat.reshape(self.limit2-self.limit1, self.nb_channel)
        elif isinstance(peaks_index, slice):
            waveforms_flat = self.stack[peaks_index]
            waveforms = np.zeros((waveforms_flat.shape[0], self.limit2-self.limit1, self.nb_channel), dtype='float64')
            for pk_index in range(waveforms_flat.shape[0]):
                waveforms[pk_index, :, :] = waveforms_flat[pk_index, :].reshape(self.limit2-self.limit1, self.nb_channel)
        else:
            waveforms = np.zeros((len(peaks_index), self.limit2-self.limit1, self.nb_channel), dtype='float64')
            for idx, pk_index in enumerate(peaks_index):
                waveforms_flat = self.stack.get_data(start=pk_index + first, stop=pk_index + first + 1, copy=True, join=True)
                waveforms[idx, :, :] = waveforms_flat.reshape(self.limit2-self.limit1, self.nb_channel)
        # print(f'get_some_waveforms( {waveforms[0, :, :]}')
        return waveforms
    
    @property
    def all_peaks(self):
        start = self._all_peaks_buffer.first_index()
        stop = self._all_peaks_buffer.index()
        return self._all_peaks_buffer.get_data(start, stop, copy=False, join=True)

    ## catalogue constructor properties
    @property
    def nb_peak(self):
        return self._all_peaks_buffer.shape[0]

    @property
    def cluster_labels(self):
        if self.clusters is not None:
            return self.clusters['cluster_label']
        else:
            return np.array([], dtype='int64')
    
    @property
    def positive_cluster_labels(self):
        return self.cluster_labels[self.cluster_labels>=0] 

    def index_of_label(self, label):
        ind = np.nonzero(self.clusters['cluster_label']==label)[0][0]
        return ind

    def recalc_cluster_info(self):
        # print(f'self.centroids_median = {self.centroids_median}')
        pass

    def compute_one_centroid(
            self, k, flush=True,
            n_spike_for_centroid=None):
        
        if n_spike_for_centroid is None:
            n_spike_for_centroid = self.n_spike_for_centroid
        
        ind = self.index_of_label(k)
        
        n_left = self.limit1
        n_right = self.limit2
        
        # waveforms not cached
        all_peaks = self.all_peaks
        # selected = np.flatnonzero(all_peaks['cluster_label'][self.some_peaks_index]==k).tolist()
        selected = np.flatnonzero(all_peaks['cluster_label']==k)
        if selected.size > n_spike_for_centroid:
            keep = np.random.choice(
                selected.size, n_spike_for_centroid, replace=False)
            selected = selected[keep]
        
        peaks_index = selected
        # peaks_index = self.some_peaks_index[selected]
        # peak_mask = np.zeros(self.nb_peak, dtype='bool')
        # peak_mask[peaks_index] = True
        # peak_sample_indexes = all_peaks[peak_mask]['index']
        
        wf = self.get_some_waveforms(
            seg_num=0,
            peaks_index=peaks_index,
            n_left=n_left, n_right=n_right,
            waveforms=None, channel_indexes=None)
        
        med = np.median(wf, axis=0)
        mad = np.median(np.abs(wf-med),axis=0)*1.4826
        '''
        print(
            f'k = {k}; ind = {ind}'
            f'median.shape = {med.shape}'
            f'mad = {mad.shape}'
            )'''

        # median, mad = mean_std(wf, axis=0)
        # to persistant arrays
        self.centroids_median[ind, :, :] = med
        self.centroids_mad[ind, :, :] = mad
        #~ self.centroids_mean[ind, :, :] = mean
        #~ self.centroids_std[ind, :, :] = std
        self.centroids_mean[ind, :, :] = 0
        self.centroids_std[ind, :, :] = 0
        
    def compute_several_centroids(self, labels, n_spike_for_centroid=None):
        # TODO make this in paralell
        for k in labels:
            self.compute_one_centroid(
                k, flush=False,
                n_spike_for_centroid=n_spike_for_centroid)
        
    def compute_all_centroid(self, n_spike_for_centroid=None):
        
        n_left = self.limit1
        n_right = self.limit2
        
        self.centroids_median = np.zeros((self.cluster_labels.size, n_right - n_left, self.nb_channel), dtype=self.source_dtype)
        self.centroids_mad = np.zeros((self.cluster_labels.size, n_right - n_left, self.nb_channel), dtype=self.source_dtype)
        self.centroids_mean = np.zeros((self.cluster_labels.size, n_right - n_left, self.nb_channel), dtype=self.source_dtype)
        self.centroids_std = np.zeros((self.cluster_labels.size, n_right - n_left, self.nb_channel), dtype=self.source_dtype)
        
        self.compute_several_centroids(self.positive_cluster_labels, n_spike_for_centroid=n_spike_for_centroid)

    def _close(self):
        pass

    def refresh_colors(self, reset=True, palette='husl', interleaved=True):
        
        labels = self.positive_cluster_labels
        
        if reset:
            n = labels.size
            if interleaved and n>1:
                n1 = np.floor(np.sqrt(n))
                n2 = np.ceil(n/n1)
                n = int(n1*n2)
                n1, n2 = int(n1), int(n2)
        else:
            n = np.sum((self.clusters['cluster_label']>=0) & (self.clusters['color']==0))

        if n>0:
            colors_int32 = get_color_palette(n, palette=palette, output='int32')
            
            if reset and interleaved and n>1:
                colors_int32 = colors_int32.reshape(n1, n2).T.flatten()
                colors_int32 = colors_int32[:labels.size]
            
            if reset:
                mask = self.clusters['cluster_label']>=0
                self.clusters['color'][mask] = colors_int32
            else:
                mask = (self.clusters['cluster_label']>=0) & (self.clusters['color']==0)
                self.clusters['color'][mask] = colors_int32
        
        #Make colors accessible by key
        self.colors = make_color_dict(self.clusters)


class RippleCatalogueController(ControllerBase):
    
    
    def __init__(self, dataio=None, chan_grp=None, parent=None):
        ControllerBase.__init__(self, parent=parent)
        
        self.dataio = dataio

        if chan_grp is None:
            chan_grp = 0
        self.chan_grp = chan_grp

        self.geometry = self.dataio.get_geometry()

        self.nb_channel = self.dataio.nb_channel
        self.channels = np.arange(self.nb_channel, dtype='int64')

        self.init_plot_attributes()

    def init_plot_attributes(self):
        self.cluster_visible = {k: True for i, k in enumerate(self.cluster_labels)}
        self.do_cluster_count()
        self.spike_selection = np.zeros(self.dataio.nb_peak, dtype='bool')
        self.spike_visible = np.ones(self.dataio.nb_peak, dtype='bool')
        self.refresh_colors(reset=False)
        self.check_plot_attributes()
    
    def check_plot_attributes(self):
        #cluster visibility
        for k in self.cluster_labels:
            if k not in self.cluster_visible:
                self.cluster_visible[k] = True
        for k in list(self.cluster_visible.keys()):
            if k not in self.cluster_labels and k>=0:
                self.cluster_visible.pop(k)
        for code in [labelcodes.LABEL_UNCLASSIFIED,]:
                if code not in self.cluster_visible:
                    self.cluster_visible[code] = True
        self.refresh_colors(reset=False)
        self.do_cluster_count()
    
    def do_cluster_count(self):
        self.cluster_count = { c['cluster_label']:c['nb_peak'] for c in self.clusters}
        self.cluster_count[labelcodes.LABEL_UNCLASSIFIED] = 0
    
    def reload_data(self):
        self.dataio.compute_all_centroid()
        self.dataio.recalc_cluster_info()
        self.init_plot_attributes()

    @property
    def spikes(self):
        return self.dataio.all_peaks.flatten()
    @property
    def all_peaks(self):
        return self.dataio.all_peaks.flatten()
    
    @property
    def clusters(self):
        return self.dataio.clusters
    
    @property
    def cluster_labels(self):
        return self.dataio.clusters['cluster_label']
    
    @property
    def positive_cluster_labels(self):
        return self.cluster_labels[self.cluster_labels>=0] 
    
    @property
    def cell_labels(self):
        return self.dataio.clusters['cell_label']
        
    @property
    def spike_index(self):
        # return self.dataio.all_peaks[:]['index']
        return self.all_peaks['index']

    @property
    def some_peaks_index(self):
        return self.dataio.some_peaks_index

    @property
    def spike_label(self):
        return self.all_peaks['cluster_label']
    
    @property
    def spike_channel(self):
        return self.all_peaks['channel']

    @property
    def spike_segment(self):
        return self.all_peaks['segment']
    
    @property
    def have_sparse_template(self):
        return False

    def get_waveform_left_right(self):
        return self.dataio.limit1, self.dataio.limit2
    
    def get_some_waveforms(
            self, seg_num, peak_sample_indexes, channel_indexes,
            peaks_index=None):
        n_left, n_right = self.get_waveform_left_right()

        waveforms = self.dataio.get_some_waveforms(
            seg_num=seg_num, chan_grp=self.chan_grp,
            peak_sample_indexes=peak_sample_indexes,
            peaks_index=peaks_index,
            n_left=n_left, n_right=n_right, channel_indexes=channel_indexes)
        return waveforms

    @property
    def info(self):
        return self.dataio.info

    def get_extremum_channel(self, label):
        if label<0:
            return None
        
        ind, = np.nonzero(self.dataio.clusters['cluster_label']==label)
        if ind.size!=1:
            return None
        ind = ind[0]
        
        extremum_channel = self.dataio.clusters['extremum_channel'][ind]
        if extremum_channel>=0:
            return extremum_channel
        else:
            return None
        
    def refresh_colors(self, reset=True, palette = 'husl'):
        self.dataio.refresh_colors(reset=reset, palette=palette)
        
        self.qcolors = {}
        for k, color in self.dataio.colors.items():
            r, g, b = color
            self.qcolors[k] = QT.QColor(r*255, g*255, b*255)

    def update_visible_spikes(self):
        visibles = np.array([k for k, v in self.cluster_visible.items() if v ])
        self.spike_visible[:] = np.in1d(self.spike_label, visibles)

    def on_cluster_visibility_changed(self):
        self.update_visible_spikes()
        ControllerBase.on_cluster_visibility_changed(self)

    def get_waveform_centroid(self, label, metric, sparse=False, channels=None):
        if label in self.dataio.clusters['cluster_label'] and self.dataio.centroids_median is not None:
            ind = self.dataio.index_of_label(label)
            attr = getattr(self.dataio, 'centroids_'+metric)
            wf = attr[ind, :, :]
            if channels is not None:
                chans = channels
                wf = wf[:, chans]
            else:
                chans = self.channels
            
            return wf, chans
        else:
            return None, None

    def get_min_max_centroids(self):
        if self.dataio.centroids_median is not None and self.dataio.centroids_median.size>0:
            wf_min = self.dataio.centroids_median.min()
            wf_max = self.dataio.centroids_median.max()
        else:
            wf_min = 0.
            wf_max = 0.
        return wf_min, wf_max
    
    @property
    def cluster_similarity(self):
        return None
   
    @property
    def cluster_ratio_similarity(self):
        return None

    def get_threshold(self):
        return 1.

    
class RippleTriggeredWindow(QT.QMainWindow):

    def __init__(self, dataio=None):
        QT.QMainWindow.__init__(self)

        self.dataio = dataio
        self.controller = RippleCatalogueController(dataio=dataio)
        #
        # self.thread = QT.QThread(parent=self)
        # self.controller.moveToThread(self.thread)
        #
        # self.traceviewer = CatalogueTraceViewer(controller=self.controller)
        self.clusterlist = OnlineClusterPeakList(controller=self.controller)
        self.peaklist = OnlinePeakList(controller=self.controller)
        self.waveformviewer = RippleWaveformViewer(controller=self.controller)
        #
        # self.pairlist = PairList(controller=self.controller)
        # self.waveformhistviewer = WaveformHistViewer(controller=self.controller)
        
        docks = {}

        docks['waveformviewer'] = QT.QDockWidget('waveformviewer',self)
        docks['waveformviewer'].setWidget(self.waveformviewer)
        self.addDockWidget(QT.Qt.RightDockWidgetArea, docks['waveformviewer'])
        #self.tabifyDockWidget(docks['ndscatter'], docks['waveformviewer'])

        '''
        docks['waveformhistviewer'] = QT.QDockWidget('waveformhistviewer',self)
        docks['waveformhistviewer'].setWidget(self.waveformhistviewer)
        self.tabifyDockWidget(docks['waveformviewer'], docks['waveformhistviewer'])
        '''

        '''docks['traceviewer'] = QT.QDockWidget('traceviewer',self)
        docks['traceviewer'].setWidget(self.traceviewer)
        #self.addDockWidget(QT.Qt.RightDockWidgetArea, docks['traceviewer'])
        self.tabifyDockWidget(docks['waveformviewer'], docks['traceviewer'])'''
        
        docks['clusterlist'] = QT.QDockWidget('clusterlist',self)
        docks['clusterlist'].setWidget(self.clusterlist)

        docks['peaklist'] = QT.QDockWidget('peaklist',self)
        docks['peaklist'].setWidget(self.peaklist)
        self.addDockWidget(QT.Qt.LeftDockWidgetArea, docks['peaklist'])
        self.splitDockWidget(docks['peaklist'], docks['clusterlist'], QT.Qt.Vertical)
        '''
        docks['pairlist'] = QT.QDockWidget('pairlist',self)
        docks['pairlist'].setWidget(self.pairlist)
        self.tabifyDockWidget(docks['pairlist'], docks['clusterlist'])
        '''
        self.create_actions()
        self.create_toolbar()

        self.speed = 1. #  Hz
        self.timer = RefreshTimer(interval=self.speed ** -1, node=self)
        self.timer.timeout.connect(self.refresh)
        for w in self.controller.views:
            self.timer.timeout.connect(w.refresh)

    def start_refresh(self):
        self.timer.start()
        # self.thread.start()
        pass
        
    def create_actions(self):
        #~ self.act_refresh = QT.QAction('Refresh', self,checkable = False, icon=QT.QIcon.fromTheme("view-refresh"))
        self.act_refresh = QT.QAction('Refresh', self,checkable = False, icon=QT.QIcon(":/view-refresh.svg"))
        self.act_refresh.triggered.connect(self.refresh_with_reload)

    def create_toolbar(self):
        self.toolbar = QT.QToolBar('Tools')
        self.toolbar.setToolButtonStyle(QT.Qt.ToolButtonTextUnderIcon)
        self.addToolBar(QT.Qt.RightToolBarArea, self.toolbar)
        self.toolbar.setIconSize(QT.QSize(60, 40))
        
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.act_refresh)

    def warn(self, title, text):
        mb = QT.QMessageBox.warning(self, title,text, QT.QMessageBox.Ok ,  QT.QMessageBox.NoButton)
    
    def refresh_with_reload(self):
        self.controller.reload_data()
        self.refresh()
    
    def refresh(self):
        # self.controller.check_plot_attributes()
        '''
        for w in self.controller.views:
            #TODO refresh only visible but need catch on visibility changed
            #~ print(w)
            #~ t1 = time.perf_counter()
            w.refresh()
            '''
        pass

    def closeEvent(self, event):
        self.timer.stop()
        # self.thread.quit()
        # self.thread.wait()
        self.controller.dataio.stop()
        self.controller.dataio.close()
        event.accept()

class DummyDataSource:

    def __init__(self, channel_names):
        self.channel_names = channel_names

    def get_channel_names(self):
        return self.channel_names