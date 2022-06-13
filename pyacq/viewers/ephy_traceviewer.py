# -*- coding: utf-8 -*-
#~ from __future__ import (unicode_literals, print_function, division, absolute_import)


import numpy as np

#~ import matplotlib.cm
#~ import matplotlib.colors

from ephyviewer.myqt import QT
import pyqtgraph as pg
import pdb
from ephyviewer.base import BaseMultiChannelViewer, Base_MultiChannel_ParamController
from ephyviewer.datasource import InMemoryAnalogSignalSource, AnalogSignalSourceWithScatter, NeoAnalogSignalSource, AnalogSignalFromNeoRawIOSource, BaseAnalogSignalSource
from ephyviewer.tools import mkCachedBrush

from ..core import (WidgetNode, register_node_type, InputStream,
        ThreadPollInput, StreamConverter)

default_params = [
    {'name': 'xsize', 'type': 'float', 'value': 3., 'step': 0.1},
    {'name': 'xratio', 'type': 'float', 'value': 0.3, 'step': 0.1, 'limits': (0,1)},
    {'name': 'ylim_max', 'type': 'float', 'value': 10.},
    {'name': 'ylim_min', 'type': 'float', 'value': -10.},
    {'name': 'scatter_size', 'type': 'float', 'value': 10.,  'limits': (0,np.inf)},
    {'name': 'scale_mode', 'type': 'list', 'value': 'real_scale',
        'values':['real_scale', 'same_for_all', 'by_channel'] },
    {'name': 'auto_scale_factor', 'type': 'float', 'value': 0.1, 'step': 0.01, 'limits': (0,np.inf)},
    {'name': 'background_color', 'type': 'color', 'value': 'k'},
    {'name': 'vline_color', 'type': 'color', 'value': '#FFFFFFAA'},
    {'name': 'label_fill_color', 'type': 'color', 'value': '#222222DD'},
    {'name': 'label_size', 'type': 'int', 'value': 8, 'limits': (1,np.inf)},
    {'name': 'display_labels', 'type': 'bool', 'value': False},
    {'name': 'display_offset', 'type': 'bool', 'value': False},
    {'name': 'antialias', 'type': 'bool', 'value': False},
    {'name': 'refresh_interval', 'type': 'int', 'value': 100, 'limits':[5, 1000]},
    {'name': 'decimate', 'type': 'int', 'value': 1, 'limits': [1, None], },
    {'name': 'decimation_method', 'type': 'list', 'value': 'min_max', 'values': ['min_max', 'mean', 'pure_decimate',  ]},
    {'name': 'line_width', 'type': 'float', 'value': 1., 'limits': (0, np.inf)},
    ]

default_by_channel_params = [
    {'name': 'color', 'type': 'color', 'value': "#55FF00"},
    {'name': 'gain', 'type': 'float', 'value': 1, 'step': 0.1, 'decimals': 8},
    {'name': 'offset', 'type': 'float', 'value': 0., 'step': 0.1},
    {'name': 'visible', 'type': 'bool', 'value': True},
    ]


class InputStreamAnalogSignalSource(BaseAnalogSignalSource):
    def __init__(self, stream):
        BaseAnalogSignalSource.__init__(self)
        #
        self.stream = stream
        self.signals = stream.buffer
        self.sample_rate = float(stream.params['sample_rate'])
        #
        t_start = 0.
        if 't_start' in stream.params:
            t_start = float(stream.params['t_start'])
        self._t_start = t_start
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
        return self.signals.get_data(i_start, i_stop, copy=False, join=True)
    
    def time_to_index(self, t):
        return int(t * self.sample_rate)

    def index_to_time(self, ind):
        return float(ind / self.sample_rate)

class TraceViewer_ParamController(Base_MultiChannel_ParamController):

    def __init__(self, parent=None, viewer=None):
        Base_MultiChannel_ParamController.__init__(self, parent=parent, viewer=viewer, with_visible=True, with_color=True)

        # raw_gains and raw_offsets are distinguished from adjustable gains and
        # offsets associated with this viewer because it makes placement of the
        # baselines and labels very easy for both raw and in-memory sources
        if isinstance(self.viewer.source, AnalogSignalFromNeoRawIOSource):
            # use raw_gains and raw_offsets from the raw source
            self.raw_gains = self.viewer.source.get_gains()
            self.raw_offsets = self.viewer.source.get_offsets()
        else:
            # use 1 and 0 for in-memory sources, which have already been scaled
            # properly
            self.raw_gains = np.ones(self.viewer.source.nb_channel)
            self.raw_offsets = np.zeros(self.viewer.source.nb_channel)

        #TODO put this somewhere

        #~ v.addWidget(QT.QLabel(self.tr('<b>Gain zoom (mouse wheel on graph):</b>'),self))
        #~ h = QT.QHBoxLayout()
        #~ v.addLayout(h)
        #~ for label, factor in [('--', 1./5.), ('-', 1./1.1), ('+', 1.1), ('++', 5.),]:
            #~ but = QT.QPushButton(label)
            #~ but.factor = factor
            #~ but.clicked.connect(self.on_but_ygain_zoom)
            #~ h.addWidget(but)

        #~ self.ygain_factor = 1.

    @property
    def selected(self):
        selected = np.ones(self.viewer.source.nb_channel, dtype=bool)
        if self.viewer.source.nb_channel>1:
            selected[:] = False
            selected[[ind.row() for ind in self.qlist.selectedIndexes()]] = True
        return selected

    @property
    def visible_channels(self):
        visible = [self.viewer.by_channel_params['ch{}'.format(i), 'visible'] for i in range(self.source.nb_channel)]
        return np.array(visible, dtype='bool')

    @property
    def gains(self):
        gains = [self.viewer.by_channel_params['ch{}'.format(i), 'gain'] for i in range(self.source.nb_channel)]
        return np.array(gains)

    @gains.setter
    def gains(self, val):
        for c, v in enumerate(val):
            self.viewer.by_channel_params['ch{}'.format(c), 'gain'] = v

    @property
    def offsets(self):
        offsets = [self.viewer.by_channel_params['ch{}'.format(i), 'offset'] for i in range(self.source.nb_channel)]
        return np.array(offsets)

    @offsets.setter
    def offsets(self, val):
        for c, v in enumerate(val):
            self.viewer.by_channel_params['ch{}'.format(c), 'offset'] = v

    @property
    def total_gains(self):
        # compute_rescale sets adjustable gains and offsets such that
        #     data_curves = (chunk * raw_gains + raw_offsets) * gains + offsets
        #                 = chunk * (raw_gains * gains) + (raw_offsets * gains + offsets)
        #                 = chunk * total_gains + total_offsets
        return self.raw_gains * self.gains

    @property
    def total_offsets(self):
        # compute_rescale sets adjustable gains and offsets such that
        #     data_curves = (chunk * raw_gains + raw_offsets) * gains + offsets
        #                 = chunk * (raw_gains * gains) + (raw_offsets * gains + offsets)
        #                 = chunk * total_gains + total_offsets
        return (self.raw_offsets * self.gains) + self.offsets

    def estimate_median_mad(self):
        # Estimates are performed on real values for both raw and in-memory
        # sources, i.e., on sigs = chunk * raw_gains + raw_offsets, where
        # raw_gains = 1 and raw_offsets = 0 for in-memory sources.

        #~ print('estimate_median_mad')
        #~ t0 = time.perf_counter()
        sigs = self.viewer.last_sigs_chunk
        assert sigs is not None, 'Need to debug this'
        #~ print(sigs)
        #~ print(sigs.shape)

        if sigs.shape[0]>1000:
            # to fast auto scale on long signal
            ind = np.random.randint(0, sigs.shape[0], size=1000)
            sigs = sigs[ind, :]

        if sigs.shape[0] > 0:
            sigs = sigs * self.raw_gains + self.raw_offsets  # calculate on real values
            self.signals_med = med = np.median(sigs, axis=0)
            self.signals_mad = np.median(np.abs(sigs-med),axis=0)*1.4826
            self.signals_min = np.min(sigs, axis=0)
            self.signals_max = np.max(sigs, axis=0)
        else:
            # dummy median/mad/...
            n = self.viewer.source.nb_channel
            self.signals_med = np.zeros(n)
            self.signals_mad = np.ones(n)
            self.signals_min = -np.ones(n)
            self.signals_max = np.ones(n)

        #~ t1 = time.perf_counter()
        #~ print('estimate_median_mad DONE', t1-t0)
        #~ print('self.signals_med', self.signals_med)

    def compute_rescale(self):
        # estimate_median_mad operates on real values, i.e., on
        # sigs = chunk * raw_gains + raw_offsets. Consequently, the gains and
        # offsets computed here will map real values to plot coords:
        #     data_curves = (chunk * raw_gains + raw_offsets) * gains + offsets
        #                 = chunk * (raw_gains * gains) + (raw_offsets * gains + offsets)
        #                 = chunk * total_gains + total_offsets

        scale_mode = self.viewer.params['scale_mode']
        #~ print('compute_rescale', scale_mode)

        self.viewer.all_params.blockSignals(True)

        gains = np.ones(self.viewer.source.nb_channel)
        offsets = np.zeros(self.viewer.source.nb_channel)
        nb_visible = np.sum(self.visible_channels)
        #~ self.ygain_factor = 1
        if self.viewer.last_sigs_chunk is not None and self.viewer.last_sigs_chunk is not []:
            self.estimate_median_mad()

            if scale_mode=='real_scale':
                self.viewer.params['ylim_min'] = np.nanmin(self.signals_min[self.visible_channels])
                self.viewer.params['ylim_max'] = np.nanmax(self.signals_max[self.visible_channels])
            else:
                if scale_mode=='same_for_all':
                    gains[self.visible_channels] = np.ones(nb_visible, dtype=float) / max(self.signals_mad[self.visible_channels]) * self.viewer.params['auto_scale_factor']
                elif scale_mode=='by_channel':
                    gains[self.visible_channels] = np.ones(nb_visible, dtype=float) / self.signals_mad[self.visible_channels] * self.viewer.params['auto_scale_factor']
                offsets[self.visible_channels] = np.arange(nb_visible)[::-1] - self.signals_med[self.visible_channels]*gains[self.visible_channels]
                self.viewer.params['ylim_min'] = -0.5
                self.viewer.params['ylim_max'] = nb_visible-0.5

        self.gains = gains
        self.offsets = offsets
        self.viewer.all_params.blockSignals(False)

    def on_channel_visibility_changed(self):
        #~ print('on_channel_visibility_changed')
        self.compute_rescale()
        self.viewer.refresh()

    def on_but_ygain_zoom(self):
        factor = self.sender().factor
        self.apply_ygain_zoom(factor)

    def apply_ygain_zoom(self, factor_ratio, chan_index=None):

        scale_mode = self.viewer.params['scale_mode']

        self.viewer.all_params.blockSignals(True)
        if scale_mode=='real_scale':
            #~ self.ygain_factor *= factor_ratio

            self.viewer.params['ylim_max'] = self.viewer.params['ylim_max']*factor_ratio
            self.viewer.params['ylim_min'] = self.viewer.params['ylim_min']*factor_ratio

            pass
            #TODO ylims
        else :
            #~ self.ygain_factor *= factor_ratio
            if not hasattr(self, 'signals_med'):
                self.estimate_median_mad()
            if scale_mode=='by_channel' and chan_index is not None:
                # factor_ratio should be applied to only the desired channel,
                # so turn the scalar factor into a vector of ones everywhere
                # except at chan_index
                factor_ratio_vector = np.ones(self.source.nb_channel)
                factor_ratio_vector[chan_index] = factor_ratio
                factor_ratio = factor_ratio_vector
            self.offsets = self.offsets + self.signals_med*self.gains * (1-factor_ratio)
            self.gains = self.gains * factor_ratio

        self.viewer.all_params.blockSignals(False)

        self.viewer.refresh()
        #~ print('apply_ygain_zoom', factor_ratio)#, 'self.ygain_factor', self.ygain_factor)

    def apply_label_drag(self, label_y, chan_index):
        self.viewer.by_channel_params['ch{}'.format(chan_index), 'offset'] = label_y - self.signals_med[chan_index]*self.gains[chan_index]


class TraceLabelItem(pg.TextItem):

    label_dragged = QT.pyqtSignal(float)
    label_ygain_zoom = QT.pyqtSignal(float)

    def __init__(self, **kwargs):
        pg.TextItem.__init__(self, **kwargs)

        self.dragOffset = None

    def mouseDragEvent(self, ev):
        '''Emit the new y-coord of the label as it is dragged'''

        if ev.button() != QT.LeftButton:
            ev.ignore()
            return
        else:
            ev.accept()

        if ev.isStart():
            # To avoid snapping the label to the mouse cursor when the drag
            # starts, we determine the offset of the position where the button
            # was first pressed down relative to the label's origin/anchor, in
            # plot coordinates
            self.dragOffset = self.mapToParent(ev.buttonDownPos()) - self.pos()

        # The new y-coord for the label is the mouse's current position during
        # the drag with the initial offset removed
        new_y = (self.mapToParent(ev.pos()) - self.dragOffset).y()
        self.label_dragged.emit(new_y)

    def wheelEvent(self, ev):
        '''Emit a yzoom factor for the associated trace'''
        if ev.modifiers() == QT.Qt.ControlModifier:
            z = 5. if ev.delta() > 0 else 1/5.
        else:
            z = 1.1 if ev.delta() > 0 else 1/1.1
        self.label_ygain_zoom.emit(z)
        ev.accept()


class TraceViewer(BaseMultiChannelViewer, WidgetNode):
    #
    _input_specs = {'signals': dict(streamtype='signals')}
    #
    _default_params = default_params
    _default_by_channel_params = default_by_channel_params

    _ControllerClass = TraceViewer_ParamController

    # request_data = QT.pyqtSignal(float, float, float, object, object, object, object)

    def __init__(self, useOpenGL=None, source=None, **kargs):
        BaseMultiChannelViewer.__init__(self, source=source, **kargs)
        WidgetNode.__init__(self, **kargs)
        #
        self.useOpenGL = useOpenGL
        self.last_sigs_chunk = None
        self._max_point = 3000
        self.last_t_min = None

    def seek(self, t):
        self.t = t

    def closeEvent(self, event):
        event.accept()
        self.close()

    def initialize_plot(self):
        self.vline = pg.InfiniteLine(angle = 90, movable = False, pen = self.params['vline_color'])
        self.vline.setZValue(1) # ensure vline is above plot elements
        self.plot.addItem(self.vline)

        self.curves = []
        self.channel_labels = []
        self.channel_offsets_line = []
        for c in range(self.source.nb_channel):
            color = self.by_channel_params['ch{}'.format(c), 'color']
            curve = pg.PlotCurveItem(
                pen='#7FFF00', downsampleMethod='peak', downsample=1,
                autoDownsample=False, clipToView=True, antialias=False)#, connect='finite')
            self.plot.addItem(curve)
            self.curves.append(curve)

            ch_name = '{}: {}'.format(c, self.source.get_channel_name(chan=c))
            label = TraceLabelItem(text=ch_name, color=color, anchor=(0, 0.5), border=None, fill=self.params['label_fill_color'])
            label.setZValue(2) # ensure labels are drawn above scatter
            font = label.textItem.font()
            font.setPointSize(self.params['label_size'])
            label.setFont(font)
            label.label_dragged.connect(lambda label_y, chan_index=c: self.params_controller.apply_label_drag(label_y, chan_index))
            label.label_ygain_zoom.connect(lambda factor_ratio, chan_index=c: self.params_controller.apply_ygain_zoom(factor_ratio, chan_index))

            self.plot.addItem(label)
            self.channel_labels.append(label)

            offset_line = pg.InfiniteLine(angle = 0, movable = False, pen = '#7FFF00')
            self.plot.addItem(offset_line)
            self.channel_offsets_line.append(offset_line)

        if self.source.with_scatter:
            self.scatter = pg.ScatterPlotItem(size=self.params['scatter_size'], pxMode = True)
            self.plot.addItem(self.scatter)

        self.viewBox.xsize_zoom.connect(self.params_controller.apply_xsize_zoom)
        self.viewBox.ygain_zoom.connect(self.params_controller.apply_ygain_zoom)

    def on_param_change(self, params=None, changes=None):
        #~ print('on_param_change')
        #track if new scale mode
        for param, change, data in changes:
            if change != 'value': continue
            if param.name()=='scale_mode':
                self.params_controller.compute_rescale()
            if param.name()=='antialias':
                for curve in self.curves:
                    curve.updateData(antialias=self.params['antialias'])
            if param.name()=='scatter_size':
                if self.source.with_scatter:
                    self.scatter.setSize(self.params['scatter_size'])
            if param.name()=='refresh_interval':
                self.timer.setInterval(self.params['refresh_interval'])
            if param.name()=='vline_color':
                self.vline.setPen(self.params['vline_color'])
            if param.name()=='label_fill_color':
                for label in self.channel_labels:
                    label.fill = pg.mkBrush(self.params['label_fill_color'])
            if param.name()=='label_size':
                for label in self.channel_labels:
                    font = label.textItem.font()
                    font.setPointSize(self.params['label_size'])
                    label.setFont(font)

    def auto_scale(self):
        #~ print('auto_scale', self.last_sigs_chunk)
        if self.last_sigs_chunk is None:
            xsize = self.params['xsize']
            xratio = self.params['xratio']
            decimation_method = self.params['decimation_method']
            #
            t_start, t_stop = self.t - xsize*xratio , self.t + xsize*(1-xratio)
            #
            visibles, = np.nonzero(self.params_controller.visible_channels)
            total_gains = self.params_controller.total_gains
            total_offsets = self.params_controller.total_offsets
            _, _, _, _, _, _,sigs_chunk, _ = self.get_data(
                self.t, t_start, t_stop, total_gains,
                total_offsets, visibles, decimation_method)
            self.last_sigs_chunk = sigs_chunk

        self.params_controller.compute_rescale()
    
    def _refresh(self):
        #~ print('TraceViewer.refresh', 't', self.t)
        xsize = self.params['xsize']
        xratio = self.params['xratio']
        decimation_method = self.params['decimation_method']
        #
        t_start, t_stop = self.t - xsize*xratio , self.t + xsize*(1-xratio)
        #
        visibles, = np.nonzero(self.params_controller.visible_channels)
        total_gains = self.params_controller.total_gains
        total_offsets = self.params_controller.total_offsets
        #
        (
            self.t, t_start, t_stop,
            visibles, dict_curves, times_curves,
            sigs_chunk, dict_scatter) = (
                self.get_data(
                    self.t, t_start, t_stop,
                    total_gains, total_offsets, visibles, decimation_method)
            )
        self.on_data_ready(
            self.t, t_start, t_stop, visibles,
            dict_curves, times_curves, sigs_chunk, dict_scatter)
        #
        # self.request_data.emit(
        #     self.t, t_start, t_stop,
        #     total_gains, total_offsets, visibles, self.params['decimation_method'])

    def _on_new_data(self, pos, data):
        xsize = self.params['xsize']
        xratio = self.params['xratio']
        t_min = self.source.index_to_time(self.source.signals.first_index())
        t_max = self.source.index_to_time(self.source.signals.index())
        # print('t_min {:.1f}, self.t {:.1f}, t_max {:.1f}'.format(t_min, self.t, t_max))
        #
        if self.parentViewer is not None:   
            self.parentViewer.navigation_toolbar.set_start_stop(
                t_min, t_max, seek=False) 
            if self.parentViewer.navigation_toolbar.play_pause_status:
                # if self.last_t_min is not None:
                #     changeInT = t_min - self.last_t_min
                #     self.t = self.t + changeInT
                #
                # self.t = t_max - xsize*(1-xratio)
                self.parentViewer.navigation_toolbar.seek(t_max - xsize*(1-xratio))
                self.last_t_min = t_min
                # print('play_pause_status: self.t = {:.2f}'.format(self.t))
        else:
            if self.t < t_min + xsize*xratio:
                # print('\t Setting self.t = {}'.format( + xsize*xratio))
                # self.t =  + xsize*xratio
                #      # 
                self.parentViewer.navigation_toolbar.seek(t_min + xsize*xratio)
            if self.t > t_max - xsize*(1-xratio):
                # print('\t Setting self.t = {}'.format(t_max - xsize*(1-xratio)))
                # self.t = t_max - xsize*(1-xratio)
                # 
                self.parentViewer.navigation_toolbar.seek(t_max - xsize*(1-xratio))
        # self._head = pos
        pass

    def on_data_ready(
            self, t, t_start, t_stop, visibles,
            dict_curves, times_curves, sigs_chunk, dict_scatter):
        #~ print('on_data_ready', t, t_start, t_stop)

        if self.t != t:
            #~ print('on_data_ready not same t')
            return

        self.graphicsview.setBackground(self.params['background_color'])

        self.last_sigs_chunk = sigs_chunk

        offsets = self.params_controller.offsets
        gains = self.params_controller.gains
        if not hasattr(self.params_controller, 'signals_med'):
            self.params_controller.estimate_median_mad()
        signals_med = self.params_controller.signals_med

        for i, c in enumerate(visibles):
            self.curves[c].show()
            self.curves[c].setData(times_curves, dict_curves[c])

            color = self.by_channel_params['ch{}'.format(c), 'color']
            self.curves[c].setPen(color, width=self.params['line_width'])

            if self.params['display_labels']:
                self.channel_labels[c].show()
                self.channel_labels[c].setPos(t_start, offsets[c] + signals_med[c]*gains[c])
                self.channel_labels[c].setColor(color)
            else:
                self.channel_labels[c].hide()

            if self.params['display_offset']:
                self.channel_offsets_line[c].show()
                self.channel_offsets_line[c].setPos(offsets[c])
                self.channel_offsets_line[c].setPen(color)
            else:
                self.channel_offsets_line[c].hide()

        for c in range(self.source.nb_channel):
            if c not in visibles:
                self.curves[c].hide()
                self.channel_labels[c].hide()
                self.channel_offsets_line[c].hide()

        if dict_scatter is not None:
            self.scatter.clear()
            all_x = []
            all_y = []
            all_brush = []
            for k, (x, y) in dict_scatter.items():
                all_x.append(x)
                all_y.append(y)

                # here we must use cached brushes to avoid issues with
                # the SymbolAtlas in pyqtgraph >= 0.11.1.
                # see https://github.com/NeuralEnsemble/ephyviewer/issues/132
                color = self.source.scatter_colors.get(k, '#FFFFFF')
                all_brush.append(np.array([mkCachedBrush(color)]*len(x)))

            if len(all_x):
                all_x = np.concatenate(all_x)
                all_y = np.concatenate(all_y)
                all_brush = np.concatenate(all_brush)
                self.scatter.setData(x=all_x, y=all_y, brush=all_brush)

        self.vline.setPos(self.t)
        self.plot.setXRange( t_start, t_stop, padding = 0.0)
        self.plot.setYRange(self.params['ylim_min'], self.params['ylim_max'], padding = 0.0)
        #~ self.graphicsview.repaint()
        return

    def get_data(
            self, t, t_start, t_stop,
            total_gains, total_offsets, visibles, decimation_method):
        #
        i_start, i_stop = self.source.time_to_index(t_start), self.source.time_to_index(t_stop) + 2
        # print(t_start, t_stop, i_start, i_stop)
        ds_ratio = (i_stop - i_start)//self._max_point + 1
        #~ print()
        #~ print('ds_ratio', ds_ratio, 'i_start i_stop', i_start, i_stop  )

        if ds_ratio>1:
            i_start = i_start - (i_start%ds_ratio)
            i_stop = i_stop - (i_stop%ds_ratio)
            #~ print('i_start, i_stop', i_start, i_stop)

        #clip it 
        ##  i_start = max(0, i_start)
        ##  i_start = min(i_start, self.source.get_length())
        ##  #
        ##  i_stop = max(0, i_stop)
        ##  i_stop = min(i_stop, self.source.get_length())
        # 
        index_padding = 10
        i_start = max(self.source.signals.first_index() + index_padding, i_start)
        i_start = min(i_start, self.source.signals.index() - index_padding)
        #
        i_stop = max(self.source.signals.first_index() + index_padding, i_stop)
        i_stop = min(i_stop, self.source.signals.index() - index_padding)
        # print(t_start, t_stop, i_start, i_stop)
        #
        if ds_ratio>1:
            #after clip
            i_start = i_start - (i_start%ds_ratio)
            i_stop = i_stop - (i_stop%ds_ratio)

        #~ print('final i_start i_stop', i_start, i_stop  )

        sigs_chunk = self.source.get_chunk(i_start=i_start, i_stop=i_stop)

        #~ print('sigs_chunk.shape', sigs_chunk.shape)
        data_curves = sigs_chunk[:, visibles].T.copy()
        if data_curves.dtype!='float32':
            data_curves = data_curves.astype('float32')

        if ds_ratio>1:
            small_size = (data_curves.shape[1]//ds_ratio)
            if decimation_method == 'min_max':
                small_size *= 2

            small_arr = np.empty((data_curves.shape[0], small_size), dtype=data_curves.dtype)

            if decimation_method == 'min_max' and data_curves.size>0:
                full_arr = data_curves.reshape(data_curves.shape[0], -1, ds_ratio)
                small_arr[:, ::2] = full_arr.max(axis=2)
                small_arr[:, 1::2] = full_arr.min(axis=2)
            elif decimation_method == 'mean' and data_curves.size>0:
                full_arr = data_curves.reshape(data_curves.shape[0], -1, ds_ratio)
                small_arr[:, :] = full_arr.mean(axis=2)
            elif decimation_method == 'pure_decimate':
                small_arr[:, :] = data_curves[:, ::ds_ratio]
            elif data_curves.size == 0:
                pass

            data_curves = small_arr

        #~ print(data_curves.shape)

        data_curves *= total_gains[visibles, None]
        data_curves += total_offsets[visibles, None]
        dict_curves = {}
        for i, c in enumerate(visibles):
            dict_curves[c] = data_curves[i, :]

        #~ print(ds_ratio)
        t_start2 = self.source.index_to_time(i_start)
        times_curves = np.arange(data_curves.shape[1], dtype='float64') # ensure high temporal precision (see issue #28)
        times_curves /= self.source.sample_rate/ds_ratio
        if ds_ratio>1 and decimation_method == 'min_max':
            times_curves /=2
        times_curves += t_start2

        dict_scatter = None
        if self.source.with_scatter:
            pass
            dict_scatter = {}
            for k in self.source.get_scatter_babels():
                x, y = [[]], [[]]
                for i, c in enumerate(visibles):
                    scatter_inds = self.source.get_scatter(i_start=i_start, i_stop=i_stop, chan=c, label=k)
                    if scatter_inds is None: continue
                    x.append((scatter_inds-i_start)/self.source.sample_rate+t_start2)
                    y.append(sigs_chunk[scatter_inds-i_start, c]*total_gains[c]+total_offsets[c])
                dict_scatter[k] = (np.concatenate(x), np.concatenate(y))

        return t, t_start, t_stop, visibles, dict_curves, times_curves, sigs_chunk, dict_scatter

    def _check_nb_channel(self):
        self.nb_channel = self.inputs['signals'].params['shape'][1]

    def _configure(
            self, with_user_dialog=True, max_xsize=60.,
            window_label='pyacq_scope0'):
        """This method is called during `Node.configure()` and must be
        reimplemented by subclasses.
        """
        #
        self.with_user_dialog = with_user_dialog
        self.max_xsize = max_xsize
        #
        self.window_label = window_label
        self.setWindowTitle(self.window_label)

    def _initialize(self, **kargs):
        """This method is called during `Node.initialize()` and must be
        reimplemented by subclasses.
        """

        #################
        self._check_nb_channel()
        assert len(self.inputs['signals'].params['shape']) == 2, 'Are you joking ?'
        
        self.sample_rate = self.inputs['signals'].params['sample_rate']
        buf_size = int(self.sample_rate * self.max_xsize)
        self.inputs['signals'].set_buffer(size=buf_size, axisorder=[1,0], double=True)
        self.source = InputStreamAnalogSignalSource(self.inputs['signals'])
        
        self.make_params()

        # useOpenGL=True eliminates the extremely poor performance associated
        # with TraceViewer's line_width > 1.0, but it also degrades overall
        # performance somewhat and is reportedly unstable
        self.set_layout(useOpenGL=self.useOpenGL)

        self.make_param_controller()

        self.viewBox.doubleclicked.connect(self.show_params_controller)

        self.initialize_plot()
        
        self.params.param('xsize').setLimits((0, np.inf))

        # traceviewer threads
        # self.thread = QT.QThread(parent=self)
        # self.datagrabber = DataGrabber(source=self.source, viewer=self)
        # self.datagrabber.moveToThread(self.thread)
        # self.thread.start()

        # self.datagrabber.data_ready.connect(self.on_data_ready)
        # self.request_data.connect(self.datagrabber.on_request_data)
        # pdb.set_trace()
        # poller
        self.poller = ThreadPollInput(
            input_stream=self.inputs['signals'], return_data=None)
        self.poller.new_data.connect(self._on_new_data)
        # timer
        # self._head = 0
        self.timer = QT.QTimer(singleShot=False, interval=100)
        self.timer.timeout.connect(self.refresh)

    def _start(self):
        # self._head = 0
        self.estimate_decimate()
        self.reset_curves_data()
        self.poller.start()
        self.timer.start()
    
    def _stop(self):
        self.poller.stop()
        self.poller.wait()
        self.timer.stop()
    
    def _close(self):
        if self.running():
            self.stop()
        if self.with_user_dialog:
            self.params_controller.close()
    
    def refresh(self):
        self._refresh()
    
    def reset_curves_data(self):
        xsize = self.params['xsize']
        decimate = self.params['decimate']
        #~ sr = self.input.params['sample_rate']
        self.full_size = int(xsize*self.sample_rate)
        self.small_size = self.full_size//decimate
        if self.small_size%2!=0:  # ensure for min_max decimate
            self.small_size -=1
        self.full_size = self.small_size*decimate
        self.t_vect = np.arange(0,self.small_size, dtype=float)/(self.sample_rate/decimate)
        self.t_vect -= self.t_vect[-1]
        # self.curves = [np.zeros((self.small_size), dtype=float) for i in range(self.nb_channel)]

    def estimate_decimate(self, nb_point=4000):
        xsize = self.params['xsize']
        self.params['decimate'] = max(int(xsize*self.sample_rate)//nb_point, 1)
    
    def check_input_specs(self):
        """This method is called during `Node.initialize()` and may be
        reimplemented by subclasses to ensure that inputs are correctly
        configured before the node is started.
        
        In case of misconfiguration, this method must raise an exception.
        """
        pass
    
    def check_output_specs(self):
        """This method is called during `Node.initialize()` and may be
        reimplemented by subclasses to ensure that outputs are correctly
        configured before the node is started.
        
        In case of misconfiguration, this method must raise an exception.
        """
        pass
    
    def after_input_connect(self, inputname):
        """This method is called when one of the Node's inputs has been
        connected.
        """
        pass
    
    def after_output_configure(self, outputname):
        """This method is called when one of the Node's outputs has been
        configured.
        
        It may be reimplemented by subclasses.
        """
        pass
    
register_node_type(TraceViewer)

