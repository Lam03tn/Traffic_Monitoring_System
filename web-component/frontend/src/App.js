import React, { useState, useEffect, useRef } from 'react';
import CameraList from './components/CameraList';
import CameraView from './components/CameraView';
import AddViolationModal from './components/AddViolationModal';
import ViolationsQuery from './components/ViolationsQuery';
import './App.css';
import { fetchCameras } from './services/cameraConfigService';

const WEBSOCKET_URL = process.env.REACT_APP_WEBSOCKET_URL || 'ws://127.0.0.1:8000';

function App() {
  const [cameras, setCameras] = useState([]);
  const [selectedCamera, setSelectedCamera] = useState(null);
  const [showViolationModal, setShowViolationModal] = useState(false);
  const [loading, setLoading] = useState(true);
  const [videoUrl, setVideoUrl] = useState(null);
  const [streamError, setStreamError] = useState(null);
  const [activeView, setActiveView] = useState('cameras');
  const wsRef = useRef(null);
  const retryCount = useRef(0);
  const maxRetries = 5;
  const wsConnectedRef = useRef(false);
  const activeCameraIdRef = useRef(null);
  const selectedCameraRef = useRef(null);
  const [videoDimensions, setVideoDimensions] = useState({ width: 0, height: 0 });
  const refreshViolationsRef = useRef(null);
  const lastProcessedFrameRef = useRef(null);

  useEffect(() => {
    selectedCameraRef.current = selectedCamera;
  }, [selectedCamera]);

  useEffect(() => {
    const loadCameras = async () => {
      try {
        setLoading(true);
        const data = await fetchCameras();
        setCameras(data);
      } catch (error) {
        console.error('Error fetching cameras:', error);
      } finally {
        setLoading(false);
      }
    };

    loadCameras();

    return () => {
      cleanupWebSocket();
    };
  }, []);

  useEffect(() => {
    if (!selectedCamera || selectedCamera.status !== 'online') return;

    if (activeCameraIdRef.current !== selectedCamera.id) {
      console.log(`Connecting to camera ${selectedCamera.id}`);
      cleanupWebSocket();
      activeCameraIdRef.current = selectedCamera.id;
      establishWebSocketConnection(selectedCamera.id);
    }
  }, [selectedCamera]);

  useEffect(() => {
    if (activeView === 'violations') {
      cleanupWebSocket();
      setVideoUrl(null);
      setSelectedCamera(null);
      setStreamError(null);
    }
  }, [activeView]);

  const cleanupWebSocket = () => {
    if (wsRef.current) {
      console.log('Cleaning up WebSocket connection');
      wsRef.current.close();
      wsRef.current = null;
      wsConnectedRef.current = false;
    }

    if (videoUrl) {
      console.log('Revoking video URL');
      URL.revokeObjectURL(videoUrl);
      setVideoUrl(null);
    }

    activeCameraIdRef.current = null;
    lastProcessedFrameRef.current = null;
  };

  const handleWebSocketMessage = (event) => {
    try {
      console.log("WebSocket message received:", event.data.substring(0, 100) + "...");
      const data = JSON.parse(event.data);

      if (data.type === 'video_frame' && data.frame) {
        console.log(`Received video frame for camera ${data.camera_id}, frame size: ${data.frame.length} bytes`);

        const currentSelectedCamera = selectedCameraRef.current;

        if (currentSelectedCamera && data.camera_id === currentSelectedCamera.id) {
          try {
            if (!lastProcessedFrameRef.current || 
                data.timestamp !== lastProcessedFrameRef.current) {
              
              lastProcessedFrameRef.current = data.timestamp;
              
              const binary = atob(data.frame);
              console.log(`Decoded binary data, length: ${binary.length}`);
              
              const bytes = new Uint8Array(binary.length);
              for (let i = 0; i < binary.length; i++) {
                bytes[i] = binary.charCodeAt(i);
              }

              const blob = new Blob([bytes], { type: 'video/mp4' });
              console.log(`Created blob: ${blob.size} bytes, type: ${blob.type}`);

              if (videoUrl) {
                URL.revokeObjectURL(videoUrl);
              }

              const url = URL.createObjectURL(blob);
              console.log(`Created object URL: ${url}`);

              setVideoUrl(url);
              setStreamError(null);
            } else {
              console.log(`Skipping duplicate frame processing for timestamp: ${data.timestamp}`);
            }
          } catch (binaryError) {
            console.error('Error processing binary data:', binaryError);
            setStreamError(`Error processing video data: ${binaryError.message}`);
          }
        } else {
          console.log(`Ignoring frame for camera ${data.camera_id}, currently selected: ${currentSelectedCamera?.id}`);
        }
      } else if (data.type === 'error') {
        console.error('Server sent error:', data.message);
        setStreamError(data.message);
      } else {
        console.log('Received non-frame message:', data.type);
      }
    } catch (error) {
      console.error(`Error parsing WebSocket message:`, error);
      setStreamError(`Error parsing message: ${error.message}`);
    }
  };

  const handleCameraSelect = (camera) => {
    if (camera.id === selectedCamera?.id) return;

    console.log(`Selecting camera: ${camera.id}`);

    // Reset videoUrl before cleanup to ensure old video is not displayed
    setVideoUrl(null);
    setSelectedCamera(camera);

    cleanupWebSocket();

    if (camera.status === 'online') {
      activeCameraIdRef.current = camera.id;
      establishWebSocketConnection(camera.id);
    } else {
      setStreamError('Camera is offline');
    }
  };

  const establishWebSocketConnection = (cameraId) => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
      wsConnectedRef.current = false;
    }

    console.log(`Establishing new WebSocket connection to camera ${cameraId}`);

    const ws = new WebSocket(`${WEBSOCKET_URL}/camera-stream/ws/stream/${cameraId}`);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log(`WebSocket connected successfully to camera ${cameraId}`);
      wsConnectedRef.current = true;
      setStreamError(null);
      retryCount.current = 0;
      activeCameraIdRef.current = cameraId;
      lastProcessedFrameRef.current = null;
    };

    ws.onmessage = handleWebSocketMessage;

    ws.onerror = (error) => {
      console.error(`WebSocket error:`, error);
      wsConnectedRef.current = false;
      setStreamError('WebSocket connection error');
    };

    ws.onclose = (event) => {
      console.log(`WebSocket closed. Code: ${event.code}, Reason: ${event.reason}`);
      wsConnectedRef.current = false;
      wsRef.current = null;

      if (activeCameraIdRef.current === cameraId && retryCount.current < maxRetries) {
        retryCount.current += 1;
        console.log(`Retrying connection (${retryCount.current}/${maxRetries})...`);
        setTimeout(() => {
          if (activeCameraIdRef.current === cameraId) {
            establishWebSocketConnection(cameraId);
          }
        }, 3000);
      } else if (retryCount.current >= maxRetries) {
        setStreamError('Max retries reached. Unable to connect to stream.');
      }
    };
  };

  const handleAddViolation = (videoDimensions) => {
    setShowViolationModal(true);
    setVideoDimensions(videoDimensions);
  };

  const closeViolationModal = () => {
    setShowViolationModal(false);
  };

  const handleRetryConnection = () => {
    setStreamError(null);
    retryCount.current = 0;
    wsConnectedRef.current = false;
    lastProcessedFrameRef.current = null;

    cleanupWebSocket();

    if (selectedCamera && selectedCamera.status === 'online') {
      establishWebSocketConnection(selectedCamera.id);
    } else {
      const initialCamera = cameras.find(camera => camera.status === 'online');
      if (initialCamera) {
        setSelectedCamera(initialCamera);
        establishWebSocketConnection(initialCamera.id);
      } else {
        setStreamError('No online cameras available');
      }
    }
  };

  return (
    <div className="app-container">
      <div className="app-navigation">
        <div 
          className={`nav-tab ${activeView === 'cameras' ? 'active' : ''}`}
          onClick={() => setActiveView('cameras')}
        >
          Camera giám sát
        </div>
        <div 
          className={`nav-tab ${activeView === 'violations' ? 'active' : ''}`}
          onClick={() => setActiveView('violations')}
        >
          Tra cứu vi phạm
        </div>
      </div>

      {activeView === 'cameras' ? (
        <div className="main-content">
          {loading ? (
            <div className="loading">Đang tải dữ liệu camera...</div>
          ) : (
            <>
              <CameraList
                cameras={cameras}
                onSelectCamera={handleCameraSelect}
                selectedCamera={selectedCamera}
              />

              <div className="camera-view-container">
                {selectedCamera ? (
                  <CameraView
                    camera={selectedCamera}
                    videoUrl={videoUrl}
                    streamError={streamError}
                    onAddViolation={handleAddViolation}
                    onRetryConnection={handleRetryConnection}
                    onRefreshViolations={refreshViolationsRef}
                  />
                ) : (
                  <div className="no-camera-selected">
                    <p>Vui lòng chọn camera để xem hình ảnh</p>
                  </div>
                )}
              </div>
            </>
          )}
        </div>
      ) : (
        <ViolationsQuery />
      )}

      {showViolationModal && selectedCamera && (
        <AddViolationModal
          camera={selectedCamera}
          onClose={closeViolationModal}
          videoDimensions={videoDimensions}
          onRefreshViolations={refreshViolationsRef}
        />
      )}
    </div>
  );
}

export default App;