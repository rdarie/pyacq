# -*- coding: utf-8 -*-
# Copyright (c) 2016, French National Center for Scientific Research (CNRS)
# Distributed under the (new) BSD License. See LICENSE for more info.

import sys
if sys.platform == 'win32':
    import ctypes
    winmm = ctypes.WinDLL('winmm')
    winmm.timeBeginPeriod(1)

import pyqtgraph
pyqtgraph.setConfigOptions(useOpenGL=True, useNumba=True)

import faulthandler
faulthandler.enable()

from .version import version as __version__
from .core import *
from .devices import *
from .viewers import *
from .dsp import *
from .rec import *
