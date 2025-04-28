import React from 'react';

const CameraList = ({ cameras, onSelectCamera, selectedCamera }) => {
  if (!cameras || cameras.length === 0) {
    return (
      <div className="camera-list">
        <div className="camera-list-header">
          <h2>Danh sách Camera</h2>
        </div>
        <div className="empty-list">Không có camera nào được tìm thấy</div>
      </div>
    );
  }

  return (
    <div className="camera-list">
      <div className="camera-list-header">
        <h2>Danh sách Camera</h2>
      </div>
      
      <ul className="camera-items">
        {cameras.map((camera) => (
          <li 
            key={camera.id} 
            className={`camera-item ${selectedCamera && selectedCamera.id === camera.id ? 'selected' : ''}`}
            onClick={() => onSelectCamera(camera)}
          >
            <div className="camera-info">
              <h3>{camera.name}</h3>
              <p>{camera.location}</p>
            </div>
            <div className="camera-status">
              <span className={`status-dot ${camera.status}`}></span>
              {camera.status === 'online' ? 'Đang hoạt động' : 'Ngoại tuyến'}
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
};

export default CameraList;