"""
PyAudio wavelet spectrogram

Streams audio data to a QTimeFreq Node, which displays a frequency spectrogram
from a Morlet continuous wavelet transform.
"""

from pyacq.devices.audio_pyaudio import PyAudio
from pyacq.viewers import QTimeFreq, QOscilloscope
from pyacq.core import create_manager, Node, register_node_type
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np


class NoiseGenerator(Node):
    """A simple example node that generates gaussian noise.
    """
    _output_specs = {
        'signals': dict(streamtype='analogsignal', dtype='float32',
        shape=(-1, 1), compression='')}

    def __init__(self, **kargs):
        Node.__init__(self, **kargs)
        self.timer = QtCore.QTimer(singleShot=False)
        self.timer.timeout.connect(self.send_data)

    def _configure(self, chunksize=100, sample_rate=1000.):
        self.chunksize = chunksize
        self.sample_rate = sample_rate
        
        self.output.spec['shape'] = (-1, 1)
        self.output.spec['sample_rate'] = sample_rate
        self.output.spec['buffer_size'] = 1000

    def _initialize(self):
        self.head = 0
        
    def _start(self):
        self.timer.start(int(1000 * self.chunksize / self.sample_rate))

    def _stop(self):
        self.timer.stop()
    
    def _close(self):
        pass
    
    def send_data(self):
        self.head += self.chunksize
        self.output.send(np.random.normal(size=(self.chunksize, 1)).astype('float32'), index=self.head)


# Not necessary for this example, but registering the node class would make it
# easier for us to instantiate this type of node in a remote process via
# Manager.create_node()
register_node_type(NoiseGenerator)

# Start Qt application
app = pg.mkQApp()

# Create a manager to spawn worker process to record and process audio
man = create_manager()
ng = man.create_nodegroup()
    
# Create a noise generator device
dev = NoiseGenerator()
dev.configure()
dev.output.configure(protocol='tcp', interface='127.0.0.1', transfermode='plaindata')
dev.initialize()

# We are only recording a single audio channel, so we create one extra 
# nodegroup for processing TFR. For multi-channel signals, create more
# nodegroups.
workers = [man.create_nodegroup()]


# Create a viewer in the local application, using the remote process for
# frequency analysis
viewer = QTimeFreq()
viewer.configure(with_user_dialog=True, nodegroup_friends=workers)
viewer.input.connect(dev.output)
viewer.initialize()
viewer.show()

viewer.params['refresh_interval'] = 100
viewer.params['timefreq', 'f_start'] = 50
viewer.params['timefreq', 'f_stop'] = 5000
viewer.params['timefreq', 'deltafreq'] = 500
viewer.by_channel_params['ch0', 'clim'] = 2500

# Create an oscilloscope node to view the noise stream
osc = QOscilloscope()
osc.configure(with_user_dialog=True)
osc.input.connect(dev.output)
osc.initialize()
osc.show()


# Start both nodes
dev.start()
osc.start()
viewer.start()


if __name__ == '__main__':
    app = QtGui.QApplication([])
    import sys
    if sys.flags.interactive == 0:
        app.exec_()
