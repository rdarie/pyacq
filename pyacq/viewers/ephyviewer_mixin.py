# -*- coding: utf-8 -*-
#~ from __future__ import (unicode_literals, print_function, division, absolute_import)


from sqlite3 import connect
import numpy as np

#~ import matplotlib.cm
#~ import matplotlib.colors
from collections import OrderedDict
from threading import Timer, Lock
from ephyviewer.myqt import QT
import pyqtgraph as pg
import pdb
import time
import atexit
import weakref
import sys
from ephyviewer.base import BaseMultiChannelViewer, Base_MultiChannel_ParamController
from ephyviewer.datasource import AnalogSignalFromNeoRawIOSource, BaseAnalogSignalSource, BaseEventAndEpoch, BaseSpikeSource
from ephyviewer.tools import mkCachedBrush
from ephyviewer.mainviewer import MainViewer
from ephyviewer.navigation import NavigationToolBar

from ..core import (Node, WidgetNode, register_node_type,
        ThreadPollInput)
from ..devices.ripple import ripple_analogsignal_types, ripple_event_types, ripple_signal_types, ripple_sample_rates, _dtype_analogsignal, _dtype_segmentDataPacket

class InputStreamEventAndEpochSource(BaseSpikeSource):
    def __init__(self, stream):
        BaseEventAndEpoch.__init__(self)
        self.stream = stream
        self._t_start = 0
        self._t_stop = 0
        channel_names = None
        if 'channel_info' in stream.params:
            channel_names = []
            for chan_idx, chan_info in enumerate(stream.params['channel_info']):
                if 'name' in chan_info:
                    channel_names.append(chan_info['name'])
                else:
                    channel_names.append('{}_{}'.format(stream.name, chan_idx))
        self.channel_names = channel_names

    @property
    def nb_channel(self):
        return self.stream.params['nb_channel']
        
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

    def get_chunk(self, chan=0,  i_start=None, i_stop=None):
        sig_chunk = self.stream.buffer.get_data(i_start, i_stop, copy=True, join=True)
        chanMask = sig_chunk['channel'] == chan
        ev_times = sig_chunk['timestamp'][chanMask] / 3e4
        return ev_times

    def get_chunk_by_time(self, chan=0,  t_start=None, t_stop=None):
        first, last = self.stream.buffer.first_index(), self.stream.buffer.index()
        sig_chunk = self.stream.buffer.get_data(first, last, copy=True, join=True)
        all_times = sig_chunk['timestamp'] / 3e4
        chanMask = (sig_chunk['channel'] == chan)
        timeMask = (all_times > t_start) & (all_times < t_stop)
        ev_times = all_times[chanMask & timeMask]
        return ev_times


class InputStreamAnalogSignalSource(BaseAnalogSignalSource):
    def __init__(self, stream):
        BaseAnalogSignalSource.__init__(self)
        #
        self.stream = stream
        self.has_custom_dtype = self.stream.params['dtype'].names is not None
        self.signals = stream.buffer
        self.sample_rate = float(stream.params['sample_rate'])
        self.reference_signal = None
        #
        self.is_t_start_adjusted = False
        t_start = 0.
        if 't_start' in stream.params:
            t_start = float(stream.params['t_start'])
        elif self.has_custom_dtype:
            i_start = self.signals.first_index()
            sig_chunk = self.signals.get_data(
                i_start, i_start + 10, copy=True, join=True)
            t = sig_chunk['timestamp'][-1, 0]
            t_start = t / 3e4 - i_start / self.sample_rate
        self._t_start = t_start
        # print('InputStreamAnalogSignalSource; t_start = {:.3f}'.format(self._t_start))
        self._t_stop = self.signals.shape[0] / self.sample_rate + t_start
        #
        channel_names = None
        if 'channel_info' in stream.params:
            channel_names = []
            for chan_idx, chan_info in enumerate(stream.params['channel_info']):
                if 'name' in chan_info:
                    channel_names.append(chan_info['name'])
                else:
                    channel_names.append('{}_{}'.format(stream.name, chan_idx))
        self.channel_names = channel_names
        return

    @property
    def nb_channel(self):
        return self.signals.shape[1]

    def get_channel_name(self, chan=0):
        return self.channel_names[chan]

    @property
    def t_start(self):
        return self._t_start

    @property
    def t_stop(self):
        return self._t_stop

    def get_length(self):
        return self.signals.shape[0]

    def get_chunk(self, i_start=None, i_stop=None):
        sig_chunk = self.signals.get_data(i_start, i_stop, copy=True, join=True)
        if self.has_custom_dtype:
            sig_chunk = sig_chunk['value']
        if self.reference_signal is not None:
            sig_chunk = sig_chunk - sig_chunk[:, self.reference_signal][:, None]
        return sig_chunk
    

class XipppyRxBuffer(Node):
    """
    A buffer for data streamed from a Ripple NIP via xipppy.
    """
    _input_specs = {
        'raw': {
            'streamtype': 'analogsignal', 'dtype': _dtype_analogsignal,
            'sample_rate': ripple_sample_rates['raw'],
            },
        'hi-res': {
            'streamtype': 'analogsignal', 'dtype': _dtype_analogsignal,
            'sample_rate': ripple_sample_rates['hi-res'],
            },
        'hifreq': {
            'streamtype': 'analogsignal', 'dtype': _dtype_analogsignal,
            'sample_rate': ripple_sample_rates['hifreq'],
            },
        'lfp': {
            'streamtype': 'analogsignal', 'dtype': _dtype_analogsignal,
            'sample_rate': ripple_sample_rates['lfp'],
            },
        'stim': {
            'streamtype': 'event', 'shape': (-1,),
            'dtype': _dtype_segmentDataPacket},
        # 'spk': {
        #     'streamtype': 'event', 'dtype': _dtype_segmentDataPacket},
            }

    def __init__(self, max_xsize=1., **kargs):
        self.max_xsize = max_xsize
        #
        self.nb_channel = {}
        self.sources = {}
        self.pollers = {}
        Node.__init__(self, **kargs)
        
    def _configure(self):
        pass

    def _check_nb_channel(self):
        for inputname in ripple_analogsignal_types:  
            self.nb_channel[inputname] = self.inputs[inputname].params['nb_channel']
            # print('self.nb_channel[{}] = {}'.format(inputname, self.nb_channel[inputname]))

    def _initialize(self):
        self._check_nb_channel()
        for inputname in ripple_analogsignal_types:
            if self.nb_channel[inputname] > 0:
                sample_rate = self.inputs[inputname].params['sample_rate']
                # set_buffer(self, size=None, double=True, axisorder=None, shmem=None, fill=None)
                bufferParams = {
                    key: self.inputs[inputname].params[key] for key in ['double', 'axisorder', 'fill']}
                bufferParams['size'] = max(
                    int(sample_rate * self.max_xsize),
                    self.inputs[inputname].params['buffer_size'])
                # bufferParams['size'] = self.inputs[inputname].params['buffer_size']
                bufferParams['shmem'] = True if (self.inputs[inputname].params['transfermode'] == 'sharedmemory') else None
                self.inputs[inputname].set_buffer(**bufferParams)
                self.sources[inputname] = InputStreamAnalogSignalSource(self.inputs[inputname])
                # There are many ways to poll for data from the input stream. In this
                # case, we will use a background thread to monitor the stream and emit
                # a Qt signal whenever data is available.
                self.pollers[inputname] = ThreadPollInput(self.inputs[inputname], return_data=True)
                # self.pollers[inputname].new_data.connect(self.data_received)
        for inputname in ripple_event_types:
            bufferParams = {
                key: self.inputs[inputname].params[key] for key in ['double', 'axisorder', 'fill']}
            bufferParams['size'] = self.inputs[inputname].params['buffer_size']
            bufferParams['shmem'] = True if (self.inputs[inputname].params['transfermode'] == 'sharedmemory') else None
            self.inputs[inputname].set_buffer(**bufferParams)
            self.sources[inputname] = InputStreamEventAndEpochSource(self.inputs[inputname])
            # There are many ways to poll for data from the input stream. In this
            # case, we will use a background thread to monitor the stream and emit
            # a Qt signal whenever data is available.
            self.pollers[inputname] = ThreadPollInput(self.inputs[inputname], return_data=True)
            self.pollers[inputname].new_data.connect(self.data_received)
    
    def _start(self):
        for inputname, poller in self.pollers.items():
            poller.start()

    def _stop(self):
        for inputname, poller in self.pollers.items():
            poller.stop()
            poller.wait()
    
    def _close(self):
        if self.running():
            self.stop()

    def data_received(self, ptr, data):
        # print("Data received: %d %s" % (ptr, data.shape))
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

        self.setSizePolicy(QT.QSizePolicy.Minimum, QT.QSizePolicy.Maximum)

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
            self.scroll_time = QT.QScrollBar(orientation=QT.Horizontal, minimum=0, maximum=1000)
            self.mainlayout.addWidget(self.scroll_time)
            if not self.is_playing:
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
            h.addWidget(QT.QFrame(frameShape=QT.QFrame.VLine, frameShadow=QT.QFrame.Sunken))

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
            if not self.is_playing:
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
        if not self.is_playing:
            self.spinbox_time.valueChanged.disconnect(self.on_spinbox_time_changed)
            self.scroll_time.valueChanged.disconnect(self.on_scroll_time_changed)
            self.is_playing = True
            self.play_pause_signal.emit(True)

    def on_stop_pause(self):
        # if play is currently enabled, we are stopping it;
        # connect these
        if self.is_playing:
            self.spinbox_time.valueChanged.connect(self.on_spinbox_time_changed)
            self.scroll_time.valueChanged.connect(self.on_scroll_time_changed)
            self.is_playing = False
            self.play_pause_signal.emit(False)

    def on_play_pause_shortcut(self):
        if self.is_playing:
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
            self.scroll_time.blockSignals(True)
            pos = int((self.t - self.t_start)/(self.t_stop - self.t_start)*1000.)
            self.scroll_time.setValue(pos)
            self.scroll_time.blockSignals(False)
            
        if refresh_spinbox and self.show_spinbox:
            self.spinbox_time.blockSignals(True)
            self.spinbox_time.setValue(t)
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
        self.timer.pause_for(500e-3)
        return

    def auto_scale(self):
        #~ print('on_xsize_changed', xsize)
        self.timer.pause_for(500e-3)
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
        self.time_reference_source = time_reference_source
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
        self.timer.start()

        self.navigation_toolbar.time_changed.connect(self.on_time_changed)
        self.navigation_toolbar.xsize_changed.connect(self.on_xsize_changed)
        self.navigation_toolbar.auto_scale_requested.connect(self.auto_scale)
        self.navigation_toolbar.speedSpin.valueChanged.connect(self.on_change_speed)
        self.load_one_setting('navigation_toolbar', self.navigation_toolbar)

    def set_time_reference_source(self, source):
        self.time_reference_source = source

    def seek(self, t):
        for name, viewer in self.viewers.items():
            viewer['widget'].seek(t)
    
    def add_view(
        self, widget, connect_seek_time=True,
        **kwargs):
        MainViewer.add_view(self, widget, **kwargs)
        if connect_seek_time:
            widget.connect_to_seek(self.seek_time)
        return
        
    def refresh(self):
        if self.time_reference_source is not None:
            t = self.navigation_toolbar.t
            source = self.time_reference_source
            t_min = source.index_to_time(source.get_first_index())
            t_max = source.index_to_time(source.get_last_index())
            nav_t_start = min(
                self.navigation_toolbar.t_start, t_min)
            nav_t_stop = max(
                self.navigation_toolbar.t_stop, t_max)
            self.navigation_toolbar.set_start_stop(
                nav_t_start, nav_t_stop, seek=False)
            if self.navigation_toolbar.is_playing:
                t = t_max - self.xsize * (1 - self.xratio)
                self.navigation_toolbar.seek(t, refresh_spinbox=False, emit=False)
            self.seek_time.emit(t)

    def on_xsize_changed(self, xsize):
        self.xsize = xsize

    def on_change_speed(self, speed):
        self.speed = speed
        self.timer.set_interval(speed ** -1)

    def closeEvent(self, event):
        self.timer.stop()
        for name, viewer in self.viewers.items():
            viewer['widget'].close()
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

    def pause_for(self, pause_duration):
        with self.lock:
            self.is_disabled = True
        time.sleep(pause_duration)
        with self.lock:
            self.is_disabled = False
        return

    def stop(self):
        with self.lock:
            self.running = False
