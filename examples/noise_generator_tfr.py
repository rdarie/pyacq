"""
PyAudio wavelet spectrogram

Streams audio data to a QTimeFreq Node, which displays a frequency spectrogram
from a Morlet continuous wavelet transform.
"""

from pyacq.devices import NoiseGenerator
from pyacq.viewers import QTimeFreq, QOscilloscope
from pyacq.core import create_manager, Node, register_node_type
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np


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
