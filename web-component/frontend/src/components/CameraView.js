import React, { useRef, useState, useEffect } from 'react';
import { getCameraActiveViolations } from '../services/cameraConfigService';
import '../css/Violation.css';

const CameraView = ({ camera, videoUrl, streamError, onAddViolation, onRetryConnection, onRefreshViolations }) => {
  const videoRef = useRef(null);
  const currentVideoUrlRef = useRef(null);
  const [videoDimensions, setVideoDimensions] = useState({ width: 0, height: 0 });
  const [activeViolations, setActiveViolations] = useState([]);
  const [loadingViolations, setLoadingViolations] = useState(false);
  const isFetching = useRef(false);

  const fetchActiveViolations = async () => {
    if (isFetching.current) return;
    isFetching.current = true;
    setLoadingViolations(true);
    try {
      const violations = await getCameraActiveViolations(camera.id);
      setActiveViolations(violations);
    } catch (error) {
      console.error('Failed to fetch active violations:', error);
    } finally {
      setLoadingViolations(false);
      isFetching.current = false;
    }
  };

  useEffect(() => {
    setVideoDimensions({ width: 0, height: 0 });
    currentVideoUrlRef.current = null;
    if (videoRef.current) {
      videoRef.current.src = ''; // Clear video source when camera changes
    }
  }, [camera.id]);

  useEffect(() => {
    if (camera.status !== 'online') return;

    fetchActiveViolations();

    return () => {
      // Cleanup (if needed)
    };
  }, [camera.id, camera.status]);

  useEffect(() => {
    if (onRefreshViolations) {
      onRefreshViolations.current = fetchActiveViolations;
    }
  }, [onRefreshViolations]);

  useEffect(() => {
    const video = videoRef.current;

    if (video && videoUrl && videoUrl !== currentVideoUrlRef.current) {
      console.log(`Updating video source to new URL`);
      video.src = videoUrl;
      currentVideoUrlRef.current = videoUrl;
    }
  }, [videoUrl]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const updateDimensions = () => {
      if (video.videoWidth && video.videoHeight) {
        setVideoDimensions({
          width: video.videoWidth,
          height: video.videoHeight,
        });
      }
    };

    video.addEventListener('loadedmetadata', updateDimensions);
    updateDimensions();

    return () => {
      video.removeEventListener('loadedmetadata', updateDimensions);
    };
  }, []);

  const getViolationDisplayName = (violationType) => {
    const names = {
      'traffic_light': 'Vượt đèn đỏ',
      'wrong_way': 'Đi ngược chiều',
      'speeding': 'Quá tốc độ',
      'illegal_parking': 'Đỗ xe trái phép'
    };
    return names[violationType] || violationType;
  };

  const renderVideo = () => {
    if (camera.status !== 'online') {
      return (
        <div className="offline-message">
          <p>Camera hiện không hoạt động</p>
        </div>
      );
    }

    if (streamError) {
      return (
        <div className="stream-error">
          <p>{streamError || 'Không thể kết nối đến luồng video. Vui lòng thử lại sau.'}</p>
          <button onClick={onRetryConnection}>Thử lại</button>
        </div>
      );
    }

    return (
      <div className="video-player">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          onError={(e) => {
            console.error('Video playback error:', e.target.error);
            const videoElement = e.target;
            console.log('Video element state:', {
              readyState: videoElement.readyState,
              networkState: videoElement.networkState,
              error: videoElement.error ? {
                code: videoElement.error.code,
                message: videoElement.error.message
              } : 'none'
            });
          }}
          onLoadedMetadata={(e) => {
            console.log('Video metadata loaded:', {
              width: e.target.videoWidth,
              height: e.target.videoHeight,
              duration: e.target.duration
            });
            if (videoRef.current) {
              setVideoDimensions({
                width: videoRef.current.videoWidth,
                height: videoRef.current.videoHeight
              });
            }
          }}
          onLoadStart={() => console.log('Video load started')}
          onPlay={() => console.log('Video playback started')}
          style={{ maxWidth: '100%', maxHeight: '100%', display: 'block' }}
        />
        {!videoUrl && (
          <div className="loading-overlay">Đang tải video từ camera...</div>
        )}
      </div>
    );
  };

  return (
    <div className="camera-view">
      <div className="camera-view-header">
        <div className="camera-details">
          <h2>{camera.name}</h2>
          <p>Vị trí: {camera.location}</p>
          <p className="camera-status-indicator">
            <span className={`status-dot ${camera.status}`}></span>
            {camera.status === 'online' ? 'Đang hoạt động' : 'Ngoại tuyến'}
          </p>
        </div>
        <div className="header-actions">
          <button
            className="add-violation-btn"
            onClick={() => onAddViolation(videoDimensions)}
            disabled={camera.status !== 'online' || !videoDimensions.width}
          >
            Thêm vi phạm
          </button>
          {loadingViolations ? (
            <span className="violation-loading">Đang tải...</span>
          ) : activeViolations.length > 0 && (
            <div className="violation-box">
              <h4>Vi phạm đang xử lý:</h4>
              <ul>
                {activeViolations.map((violation, index) => (
                  <li key={index}>{getViolationDisplayName(violation)}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>

      <div className="video-container">
        {renderVideo()}
      </div>
    </div>
  );
};

export default CameraView;