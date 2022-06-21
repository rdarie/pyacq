from pyacq.core import Node, register_node_type
from pyacq.core.tools import ThreadPollInput

class StreamMonitor(Node):
    """
    Monitors activity on an input stream and prints details about packets
    received.
    """
    _input_specs = {'signals': {}}
    
    def __init__(self, **kargs):
        Node.__init__(self, **kargs)
    
    def _configure(self):
        pass

    def _initialize(self):
        # There are many ways to poll for data from the input stream. In this
        # case, we will use a background thread to monitor the stream and emit
        # a Qt signal whenever data is available.
        self.poller = ThreadPollInput(self.input, return_data=True)
        self.poller.new_data.connect(self.data_received)
        
    def _start(self):
        self.poller.start()
        
    def data_received(self, ptr, data):
        print("Data received: %d %s %s" % (ptr, data.shape, data.dtype))

register_node_type(StreamMonitor)