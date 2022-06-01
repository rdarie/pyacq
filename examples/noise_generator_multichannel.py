# -*- coding: utf-8 -*-
# Copyright (c) 2016, French National Center for Scientific Research (CNRS)
# Distributed under the (new) BSD License. See LICENSE for more info.
"""
Noise generator node

Simple example of a custom Node class that generates a stream of random
values. 

"""
import numpy as np

from pyacq.core import Node, register_node_type
from pyqtgraph.Qt import QtCore, QtGui
from pyacq.devices import NoiseGenerator

if __name__ == '__main__':
    from pyacq.viewers import QOscilloscope
    app = QtGui.QApplication([])
    
    # Create a noise generator node
    ng = NoiseGenerator()
    ng.configure(num_chans=10)
    ng.output.configure(protocol='inproc', transfermode='sharedmem')
    ng.initialize()
    
    # Create an oscilloscope node to view the noise stream
    osc = QOscilloscope()
    osc.configure(with_user_dialog=True)
    osc.input.connect(ng.output)
    osc.initialize()
    osc.show()

    # start both nodes
    osc.start()
    ng.start()

    # start the app
    app.exec_()