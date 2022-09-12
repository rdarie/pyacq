import pdb, traceback
import warnings

from pyacq.core import Node, OutputStream, RPCClient, register_node_type
from pyacq.core.tools import ThreadPollInput, weakref, make_dtype
from ephyviewer.myqt import QT
from pyqtgraph.util.mutex import Mutex
import numpy as np
import pandas as pd

from MOCK_vicon_dssdk.vicon_dssdk import ViconDataStream

vicon_analogsignal_types = [
    'segments',
    'markers',
    'unlabeled_markers',
    'marker_rays',
    'devices',
    'centroids'
]

vicon_enable_fun_names = {
    'segments': "EnableSegmentData",
    'markers': "EnableMarkerData",
    'unlabeled_markers': "EnableUnlabeledMarkerData",
    'marker_rays': "EnableMarkerRayData",
    'devices': "EnableDeviceData",
    'centroids': "EnableCentroidData"
}
vicon_signal_names = {
    'segments': "Segment",
    'markers': "Marker",
    'unlabeled_markers': "UnlabeledMarker",
    'marker_rays': "MarkerRay",
    'devices': "Device",
    'centroids': "Centroid"
}
vicon_signal_names_implemented = [
    'markers', 'devices'
]

_dtype_vicon_timecode = [
    ('hours', 'int64', (1,)),
    ('minutes', 'int64', (1,)),
    ('seconds', 'int64', (1,)),
    ('frames', 'int64', (1,)),
    ('subframe', 'int64', (1,)),
    ('fieldFlag', 'int64', (1,)),
    ('standard', 'int64', (1,)),
    ('subFramesPerFrame', 'int64', (1,)),
    ('userBits', 'int64', (1,)),
    ]

_dtype_vicon_marker_position = [
    ('x', 'float64', (1,)),
    ('y', 'float64', (1,)),
    ('z', 'float64', (1,)),
    ('occluded', 'bool', (1,))
    ]
'''
    Vicon DataStream SDK client.`

    This class manages connection to the Vicon tracking data.
    Vicon data types:

    SegmentData (variant LightweightSegmentData)
    MarkerData
    UnlabelledMarkerData
    MarkerRayData
    CentroidData
    DeviceData

    StreamMode( Enum ):
        EClientPull = 0
        EClientPullPreFetch = 1
        EServerPush = 2
    
    ## more info on line 653 of ViconDataStream.py in the DatastreamSDK

    class TimecodeStandard( Enum ):
        ETimecodeStandardNone = 0
        EPAL = 1
        ENTSC = 2
        ENTSCDrop = 3
        EFilm = 4
        ENTSCFilm = 5
        EATSC = 6

    ## more info on line 791 of ViconDataStream.py in the DatastreamSDK

    Vicon DataStream SDK retiming client.`

      ===========================
      Intended uses
      -------------
      The Vicon DataStream re-timing client provides calls to obtain subject data from the 
      DataStream with minimal latency and temporal jitter.
'''

class Vicon(Node):
    """
    A Node is the basic element for generating and processing data streams
    in pyacq. 
    
    Nodes may be used to interact with devices, generate data, store data, 
    perform computations, or display user interfaces. Each node may have multiple
    input and output streams that connect to other nodes. For example::
    
       [ data acquisition node ] -> [ processing node ] -> [ display node ]
                                                        -> [ recording node ]
    
    An application may directly create and connect the Nodes it needs, or it
    may use a Manager to create a network of nodes distributed across multiple
    processes or machines.
    
    The order of operations when creating and operating a node is very important:
    
    1. Instantiate the node directly or remotely using `NodeGroup.create_node`.
    2. Call `Node.configure(...)` to set global parameters such as sample rate,
       channel selections, etc.
    3. Connect inputs to their sources (if applicable):
       `Node.inputs['input_name'].connect(other_node.outpouts['output_name'])`
    4. Configure outputs: `Node.outputs['output_name'].configure(...)`
    5. Call `Node.initialize()`, which will verify input/output settings, 
       allocate memory, prepare devices, etc.
    6. Call `Node.start()` and `Node.stop()` to begin/end reading from input 
       streams and writing to output streams. These may be called multiple times.
    7. Close the node with `Node.close()`. If the node was created remotely, 
       this is handled by the NodeGroup to which it belongs.
    
    Notes
    -----
    
    For convenience, if a Node has only 1 input or 1 output:
    
    * `Node.inputs['input_name']` can be written `Node.input`
    * `Node.outputs['output_name']` can be written `Node.output`
    
    When there are several outputs or inputs, this shorthand is not permitted.
    
    The state of a Node can be requested using thread-safe methods:
    
    * `Node.running()`
    * `Node.configured()`
    * `Node.initialized()`
    """
    _output_specs = {}

    _default_axis_map = (
        ViconDataStream.Client.AxisMapping.EForward, ViconDataStream.Client.AxisMapping.ELeft, ViconDataStream.Client.AxisMapping.EUp
        )
    
    def __init__(
            self, name='', parent=None, requested_signal_types=None, verbose=False):
        if requested_signal_types is None:
            # enable all by default
            self.requested_signal_types = vicon_analogsignal_types
        else:
            self.requested_signal_types = requested_signal_types
            pop_these_types = [
                signal_type
                for signal_type in self._output_specs
                if signal_type not in self.requested_signal_types
                ]
            for signal_type in pop_these_types:
                self._output_specs.pop(signal_type, None)
        self.verbose = verbose
        Node.__init__(self, name=name, parent=parent)
    
    def _configure(
        self, ip_address="localhost", port="",
        buffer_size=1, stream_mode=None, axis_map=None, **kargs):
        """This method is called during `Node.configure()` and must be
        reimplemented by subclasses.
        """
        self.vicon_hostname = f"{ip_address}:{port}"
        self.buffer_size = buffer_size
        self.stream_mode = stream_mode
        self.axis_map = self._default_axis_map if axis_map is None else axis_map
        
        try:
            self.vicon_client = ViconDataStream.Client()
            self.vicon_client.Connect(self.vicon_hostname)
            # Check the version
            print('ViconDataStream.Client()\n\nVersion', self.vicon_client.GetVersion())
            # Check setting the buffer size works
            self.vicon_client.SetBufferSize(self.buffer_size)
            # enable requested data streams
            for signal_name in self.requested_signal_types:
                fun_name = f"Enable{vicon_signal_names[signal_name]}Data"
                enabling_fun = getattr(self.vicon_client, fun_name)
                # e.g. enabling_fun = client.EnableSegmentData
                enabling_fun()
                # Report whether the data type has been enabled
                checking_fun_name = f"Is{vicon_signal_names[signal_name]}DataEnabled"
                checking_fun = getattr(self.vicon_client, checking_fun_name)
                print(f"{checking_fun_name}() = {checking_fun()}")
            # connect once to establish current frame
            HasFrame = False
            while not HasFrame:
                try:
                    self.vicon_client.GetFrame()
                    HasFrame = True
                except ViconDataStream.DataStreamException as e:
                    self.vicon_client.GetFrame()
            #
            self.vicon_client.SetStreamMode(self.stream_mode)
            print(
                f'Get Frame {self.stream_mode}',
                self.vicon_client.GetFrame(), self.vicon_client.GetFrameNumber())
            
            self.vicon_client.SetAxisMapping(*self.axis_map)
            self.xAxis, self.yAxis, self.zAxis = self.vicon_client.GetAxisMapping()
            if self.verbose:
                print( 'X Axis', self.xAxis, 'Y Axis', self.yAxis, 'Z Axis', self.zAxis )

            self.outputs = {}
            self.output_specs = {}
            sample_rate = self.vicon_client.GetFrameRate()
            if 'markers' in self.requested_signal_types:
                for subjectName in self.vicon_client.GetSubjectNames():
                    for markerName, parentSegment in self.vicon_client.GetMarkerNames(subjectName):
                        this_spec = {
                            'streamtype': 'analogsignal',
                            'sample_rate': sample_rate,
                            'segment': parentSegment,
                            'vicon_type': 'marker',
                            'nb_channel': 1, # custom dtype of (x, y, z and occluded flag)
                            'shape': (-1, 1),
                            'dtype': _dtype_vicon_marker_position,
                            'buffer_size': 1000
                            }
                        self.output_specs[markerName] = this_spec
                        self.outputs[markerName] = OutputStream(spec=this_spec, node=self, name=markerName)
            if 'devices' in self.requested_signal_types:
                devDetailsDict = {}
                for deviceName, deviceType in self.vicon_client.GetDeviceNames():
                    output_details = self.vicon_client.GetDeviceOutputDetails(deviceName)
                    devDetailsDict[(deviceName, deviceType)] = pd.DataFrame(output_details, columns=['outputName', 'componentName', 'unit'])
                self._device_details = pd.concat(devDetailsDict, names=['deviceName', 'deviceType', 'oIdx']).reset_index().drop(columns=['oIdx'])
                group_name_list = ['deviceName', 'deviceType', 'outputName']
                for outputNameList, group in self._device_details.groupby(group_name_list, sort=False):
                    deviceName, deviceType, outputName = outputNameList
                    thisName = f'{deviceName} - {outputName}'
                    columnsToSave = [cN for cN in group.columns if cN not in group_name_list]
                    this_spec = {
                        'streamtype': 'analogsignal',
                        'sample_rate': sample_rate,
                        'vicon_type': 'device',
                        'device_type': deviceType,
                        'output_details': group.loc[:, columnsToSave],
                        'nb_channel': group.shape[0],
                        'shape': (-1,  group.shape[0]),
                        'dtype': float,
                        'buffer_size': 1000
                        }
                    self.output_specs[thisName] = this_spec
                    self.outputs[thisName] = OutputStream(spec=this_spec, node=self, name=thisName)
        except ViconDataStream.DataStreamException as e:
            print('Handled data stream error', e)


    def _initialize(self, **kargs):
        """
            This method is called during `Node.initialize()` and must be
            reimplemented by subclasses.
            """
        self.thread = ViconClientThread(self, parent=None)
    
    def _start(self):
        """This method is called during `Node.start()` and must be
        reimplemented by subclasses.
        """
        self.thread.start()
    
    def _stop(self):
        """This method is called during `Node.stop()` and must be
        reimplemented by subclasses.
        """
        self.thread.stop()
        self.thread.wait()

    def _close(self, **kargs):
        """This method is called during `Node.close()` and must be
        reimplemented by subclasses.
        """
        self.vicon_client.Disconnect()
    
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
        
        It may be reimplemented by subclasses.
        """
        pass
    
    def after_output_configure(self, outputname):
        """This method is called when one of the Node's outputs has been
        configured.
        
        It may be reimplemented by subclasses.
        """
        pass
    
    def get_timecode(self):
        raw_timecode = self.vicon_client.GetTimecode()
        timecode = np.array([raw_timecode,], dtype=_dtype_vicon_timecode)
        if self.verbose:
            print(
                ('Timecode:', timecode['hours'], 'hours', timecode['minutes'], 'minutes', timecode['seconds'], 'seconds', timecode['frames'], 
                'frames', timecode['subframe'], 'sub frame', timecode['fieldFlag'], 'field flag', 
                timecode['standard'], 'standard', timecode['subFramesPerFrame'], 'sub frames per frame', timecode['userBits'], 'user bits'))
        return timecode

    def get_latency(self):
        latency_total = self.vicon_client.GetLatencyTotal()
        latency_samples = self.vicon_client.GetLatencySamples()
        if self.verbose:
            print( 'Total Latency', latency_total )
            print( 'Latency Samples' )
            for sampleName, sampleValue in latency_samples.items():
                print( sampleName, sampleValue )
        return latency_total, latency_samples

class ViconClientThread(QT.QThread):
    """
    Vicon thread that grab continuous data.
    """
    def __init__(self, node, parent=None):
        QT.QThread.__init__(self, parent=parent)
        self.node = node

        self.vicon_client = self.node.vicon_client
        self.requested_signal_types = self.node.requested_signal_types
        self.verbose = self.node.verbose

        self.lock = Mutex()
        self.running = False

    def inspect_segments(self, subjectName):
        segmentNames = self.vicon_client.GetSegmentNames(subjectName)
        for segmentName in segmentNames:
            segmentChildren = self.vicon_client.GetSegmentChildren( subjectName, segmentName )
            for child in segmentChildren:
                try:
                    print( child, 'has parent', self.vicon_client.GetSegmentParentName( subjectName, segmentName ) )
                except ViconDataStream.DataStreamException as e:
                    print( 'Error getting parent segment', e )
            print( segmentName, 'has static translation', self.vicon_client.GetSegmentStaticTranslation( subjectName, segmentName ) )
            print( segmentName, 'has static rotation( helical )', self.vicon_client.GetSegmentStaticRotationHelical( subjectName, segmentName ) )               
            print( segmentName, 'has static rotation( EulerXYZ )', self.vicon_client.GetSegmentStaticRotationEulerXYZ( subjectName, segmentName ) )              
            print( segmentName, 'has static rotation( Quaternion )', self.vicon_client.GetSegmentStaticRotationQuaternion( subjectName, segmentName ) )               
            print( segmentName, 'has static rotation( Matrix )', self.vicon_client.GetSegmentStaticRotationMatrix( subjectName, segmentName ) )
            try:
                print( segmentName, 'has static scale', self.vicon_client.GetSegmentStaticScale( subjectName, segmentName ) )
            except ViconDataStream.DataStreamException as e:
                print( 'Scale Error', e )               
            print( segmentName, 'has global translation', self.vicon_client.GetSegmentGlobalTranslation( subjectName, segmentName ) )
            print( segmentName, 'has global rotation( helical )', self.vicon_client.GetSegmentGlobalRotationHelical( subjectName, segmentName ) )               
            print( segmentName, 'has global rotation( EulerXYZ )', self.vicon_client.GetSegmentGlobalRotationEulerXYZ( subjectName, segmentName ) )               
            print( segmentName, 'has global rotation( Quaternion )', self.vicon_client.GetSegmentGlobalRotationQuaternion( subjectName, segmentName ) )               
            print( segmentName, 'has global rotation( Matrix )', self.vicon_client.GetSegmentGlobalRotationMatrix( subjectName, segmentName ) )
            print( segmentName, 'has local translation', self.vicon_client.GetSegmentLocalTranslation( subjectName, segmentName ) )
            print( segmentName, 'has local rotation( helical )', self.vicon_client.GetSegmentLocalRotationHelical( subjectName, segmentName ) )               
            print( segmentName, 'has local rotation( EulerXYZ )', self.vicon_client.GetSegmentLocalRotationEulerXYZ( subjectName, segmentName ) )               
            print( segmentName, 'has local rotation( Quaternion )', self.vicon_client.GetSegmentLocalRotationQuaternion( subjectName, segmentName ) )               
            print( segmentName, 'has local rotation( Matrix )', self.vicon_client.GetSegmentLocalRotationMatrix( subjectName, segmentName ) )
        try:
            print( 'Object Quality', self.vicon_client.GetObjectQuality( subjectName ) )
        except ViconDataStream.DataStreamException as e:
                print( 'Not present', e )

    def inspect_markers(self, subjectName):
        markerNames = self.vicon_client.GetMarkerNames( subjectName )
        for markerName, parentSegment in markerNames:
            print( markerName, 'has parent segment', parentSegment, 'position', self.vicon_client.GetMarkerGlobalTranslation( subjectName, markerName ) )
            rayAssignments = self.vicon_client.GetMarkerRayAssignments( subjectName, markerName )
            if len( rayAssignments ) == 0:
                print( 'No ray assignments for', markerName )
            else:
                for cameraId, centroidIndex in rayAssignments:
                    print( 'Ray from', cameraId, 'centroid', centroidIndex )

    def inspect_unlabeled_markers(self):
        unlabeledMarkers = self.vicon_client.GetUnlabeledMarkers()
        for markerPos, trajID in unlabeledMarkers:
            print( 'Unlabeled Marker at', markerPos, 'with trajID', trajID )
            
    def inspect_labeled_markers(self):
        labeledMarkers = self.vicon_client.GetLabeledMarkers()
        for markerPos, trajID in labeledMarkers:
            print( 'Labeled Marker at', markerPos, 'with trajID', trajID )

    def read_forceplate(self, plate):
        # e.g. for plate in client.GetForcePlates():
        forceVectorData = self.vicon_client.GetForceVector( plate )
        momentVectorData = self.vicon_client.GetMomentVector( plate )
        copData = self.vicon_client.GetCentreOfPressure( plate )
        globalForceVectorData = self.vicon_client.GetGlobalForceVector( plate )
        globalMomentVectorData = self.vicon_client.GetGlobalMomentVector( plate )
        globalCopData = self.vicon_client.GetGlobalCentreOfPressure( plate )
        try:
            analogData = self.vicon_client.GetAnalogChannelVoltage( plate )
        except ViconDataStream.DataStreamException as e:
            print( 'Failed getting analog channel voltages' )
            analogData = None
        raw_forceplate_reading = (
            forceVectorData, momentVectorData, copData, 
            globalForceVectorData, globalMomentVectorData, globalCopData, 
            analogData
            )
        return raw_forceplate_reading

    def read_eye_tracker(self, eyeTracker):
        # e.g. for eyeTracker in client.GetEyeTrackers():
        position, position_occluded = self.vicon_client.GetEyeTrackerGlobalPosition( eyeTracker )
        gaze, gaze_occluded = self.vicon_client.GetEyeTrackerGlobalGazeVector( eyeTracker )
        if self.verbose:
            print( 'Eye Tracker', gaze, position )
        return (position, position_occluded, gaze, gaze_occluded)

    def inspect_camera(self, camera):
        # e.g. for camera in client.GetCameraNames():
        cameraId = self.vicon_client.GetCameraID( camera )
        userId = self.vicon_client.GetCameraUserID( camera )
        type = self.vicon_client.GetCameraType( camera )
        displayName = self.vicon_client.GetCameraDisplayName( camera )
        resX, resY = self.vicon_client.GetCameraResolution( camera )
        isVideo = self.vicon_client.GetIsVideoCamera( camera )
        centroids = self.vicon_client.GetCentroids( camera )
        if self.verbose:
            print( cameraId, userId, type, displayName, resX, resY, isVideo )
            for centroid, radius, weight in centroids:
                print( centroid, radius, weight )
        raw_camera_info = (
            cameraId, userId, type, displayName,
            resX, resY, isVideo, centroids
            )
        return raw_camera_info

    def run(self):
        with self.lock:
            self.running = True
        #
        while True:
            with self.lock:
                if not self.running:
                    break
            try:
                self.vicon_client.GetFrame()
                #
                subjectNames = self.vicon_client.GetSubjectNames()
                for subjectIdx, subjectName in enumerate(subjectNames):
                    if 'markers' in self.requested_signal_types:
                        markerNames = [mn[0] for mn in self.vicon_client.GetMarkerNames(subjectName)]
                        for markerIdx, markerName in enumerate(markerNames):
                            raw_marker_position, occluded = self.vicon_client.GetMarkerGlobalTranslation(subjectName, markerName)
                            marker_position = np.array(
                                [(*raw_marker_position, occluded),], dtype=_dtype_vicon_marker_position)
                            self.node.outputs[markerName].send(marker_position[np.newaxis])
                if 'devices' in self.requested_signal_types:
                    group_name_list = ['deviceName', 'outputName', 'deviceType']
                    for (devName, outName, devType), group in self.node._device_details.groupby(group_name_list, sort=False):
                        this_set = []
                        for compName in group['componentName']:
                            values, occluded = self.vicon_client.GetDeviceOutputValues(devName, outName, compName)
                            this_set.append(np.asarray(values)[:, np.newaxis])
                        thisName = f"{devName} - {outName}"
                        data = np.concatenate(this_set, axis=1)
                        self.node.outputs[thisName].send(data)
                # WIP alternative marker workflow based on trajID
                '''
                if 'markers' in self.requested_signal_types:
                    labeledMarkers = self.vicon_client.GetLabeledMarkers()
                    if self.verbose:
                        for markerPos, trajID in labeledMarkers:
                            print( 'Labeled Marker at', markerPos, 'with trajID', trajID )
                if 'unlabeled_markers' in self.requested_signal_types:
                    unlabeledMarkers = self.vicon_client.GetUnlabeledMarkers()
                    if self.verbose:
                        for markerPos, trajID in unlabeledMarkers:
                            print( 'Unlabeled Marker at', markerPos, 'with trajID', trajID )'''
            except ViconDataStream.DataStreamException as e:
                print( 'Handled data stream error', e)
        
    def stop(self):
        with self.lock:
            self.running = False

class ViconRetimingClientThread(QT.QThread):
    """
    Vicon thread that grab continuous data.
    """
    def __init__(self, node, parent=None):
        QT.QThread.__init__(self, parent=parent)
        self.node = node

        self.lock = Mutex()
        self.running = False
        
    def run(self):
        with self.lock:
            self.running = True
        
        '''stream = self.node.outputs['aichannels']
        ai_channels = np.array(self.node.ai_channels, dtype='uint16')
        trialcont = self.node.trialcont
        ai_buffer = self.node.ai_buffer
        nInstance = self.node.nInstance
        nb_channel = self.node.nb_channel
        cbSdk = self.node.cbSdk
        
        n = 0
        next_timestamp = None
        first_buffer = True
        while True:
            with self.lock:
                if not self.running:
                    break
            
            # probe buffer size for all channel
            cbSdk.InitTrialData(nInstance, 1, None, ctypes.byref(trialcont), None, None)
            
            if trialcont.count==0:
                time.sleep(0.003) # a buffer is 10 ms sleep a third
                continue
            
            # samples for some channels
            num_samples = np.ctypeslib.as_array(trialcont.num_samples)
            
            # it is a personnal guess when some channel are less than 300 sample 
            # then we must waiyt for a while because some packet are not arrived yet.
            if num_samples[0] < 300:
                # it is too early!!!!!!
                time.sleep(0.003)
                continue
            
            # this really get the buffer
            cbSdk.GetTrialData(nInstance, 1, None, ctypes.byref(trialcont), None, None)
            
            if first_buffer:
                # trash the first buffer because it is a very big one
                first_buffer = False
                continue
            
            num_samples = np.ctypeslib.as_array(trialcont.num_samples)
            sample_rates = np.ctypeslib.as_array(trialcont.sample_rates)
            
            # maybe this would be safe : to be check
            # assert np.all(num_samples[0]==num_samples)
            
            num_sample = num_samples[0]
            
            # here it is a bit tricky because when can received unwanted 
            # channel so we need to align internal buffer to output buffer
            # with a mask
            channels_in_buffer = np.ctypeslib.as_array(trialcont.chan)
            channel_mask = np.in1d(channels_in_buffer, ai_channels)
            channel_mask &= (sample_rates==30000)
            
            num_samples = num_samples[channel_mask]
            
            # 2 case
            if np.sum(channel_mask)==nb_channel:
                # all channel are in the buffer: easy just a transpose
                data = ai_buffer[channel_mask, : num_sample].T.copy()
            else:
                #some are not because not configured accordingly in "central"
                data = np.zeros((num_sample, nb_channel), dtype='int16')
                outbuffer_channel_mask = np.in1d(ai_channels, channels_in_buffer[channel_mask])
                data[:, outbuffer_channel_mask] = ai_buffer[channel_mask, : num_sample].T
                
            n += data.shape[0]
            stream.send(data, index=n)
            
            # this is for checking missing packet: disable at the moment
            # if next_timestamp is not None:
            #    pass
            # next_timestamp = trialcont.time + num_sample
            
            # be nice
            time.sleep(0)'''

    def stop(self):
        with self.lock:
            self.running = False


register_node_type(Vicon)
