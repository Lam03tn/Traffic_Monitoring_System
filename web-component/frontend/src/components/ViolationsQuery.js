import React, { useState, useEffect } from 'react';
import { fetchViolationsByDate, fetchViolationsByStatus, updateViolationStatus, fetchViolationVideo, fetchViolationImage } from '../services/violationService';
import '../css/ViolationsQuery.css';

const ViolationsQuery = () => {
  const [queryType, setQueryType] = useState('date');
  const [dateQuery, setDateQuery] = useState(new Date().toISOString().split('T')[0]);
  const [statusQuery, setStatusQuery] = useState('pending');
  const [violations, setViolations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showDetails, setShowDetails] = useState({});
  const [currentPage, setCurrentPage] = useState(1);
  const [evidence, setEvidence] = useState({});
  const [modalOpen, setModalOpen] = useState(false);
  const [selectedViolationId, setSelectedViolationId] = useState(null);
  const [evidenceType, setEvidenceType] = useState(null);
  const [pendingStatusChanges, setPendingStatusChanges] = useState({});
  const [sortConfig, setSortConfig] = useState({ key: 'violation_time', direction: 'desc' });
  const violationsPerPage = 10;

  const statusOptions = [
    { value: 'pending', label: 'Chờ xử lý' },
    { value: 'processed', label: 'Đã xử lý' },
    { value: 'false_positive', label: 'Cảnh báo sai' },
    { value: 'verified', label: 'Đã xác nhận' }
  ];

  const violationTypes = {
    'traffic_light': 'Vượt đèn đỏ',
    'wrong_way': 'Đi ngược chiều',
    'speeding': 'Vượt tốc độ',
    'illegal_parking': 'Đỗ xe trái phép',
    'other': 'Vi phạm khác'
  };

  const handleQuerySubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setViolations([]);
    setCurrentPage(1);
    
    try {
      let result;
      if (queryType === 'date') {
        result = await fetchViolationsByDate(dateQuery);
      } else if (queryType === 'status') {
        result = await fetchViolationsByStatus(statusQuery);
      }
      
      setViolations(result);
    } catch (err) {
      console.error('Error fetching violations:', err);
      setError('Không thể truy vấn dữ liệu vi phạm. Vui lòng thử lại sau.');
    } finally {
      setLoading(false);
    }
  };

  const fetchEvidence = async (violation) => {
    try {
      const date = new Date(violation.violation_time);
      const timestamp = `${date.getFullYear()}${String(date.getMonth() + 1).padStart(2, '0')}${String(date.getDate()).padStart(2, '0')}_${String(date.getHours()).padStart(2, '0')}${String(date.getMinutes()).padStart(2, '0')}${String(date.getSeconds()).padStart(2, '0')}`;
      const camera_id = violation.camera_id;
      const violation_type = violation.violation_type;

      const videoBlob = await fetchViolationVideo(violation_type, camera_id, timestamp);
      const imageBlob = await fetchViolationImage(violation_type, camera_id, `${timestamp}_1`);
      const imageBlobBefore = await fetchViolationImage(violation_type, camera_id, `${timestamp}_2`);
      
      let imageBlobPlate = null;
      try {
        imageBlobPlate = await fetchViolationImage(violation_type, camera_id, `${timestamp}_3`);
      } catch (plateErr) {
        console.warn(`License plate image (${timestamp}_3.jpg) not found for violation ${violation.violation_id}: ${plateErr.message}`);
      }

      const videoUrl = URL.createObjectURL(videoBlob);
      const imageUrl = URL.createObjectURL(imageBlob);
      const imageUrlBefore = URL.createObjectURL(imageBlobBefore);
      const imageUrlPlate = imageBlobPlate ? URL.createObjectURL(imageBlobPlate) : null;

      setEvidence(prev => ({
        ...prev,
        [violation.violation_id]: { 
          videoUrl, 
          imageUrl, 
          imageUrlBefore, 
          imageUrlPlate 
        }
      }));
    } catch (err) {
      console.error('Error fetching evidence:', err);
      setError(err.message || 'Không thể tải bằng chứng vi phạm.');
    }
  };

  const toggleDetails = async (violationId) => {
    setShowDetails(prev => ({
      ...prev,
      [violationId]: !prev[violationId]
    }));
  };

  const openEvidenceModal = (violationId, type) => {
    setSelectedViolationId(violationId);
    setEvidenceType(type);
    setModalOpen(true);

    const violation = violations.find(v => v.violation_id === violationId);
    if (violation && !evidence[violationId]) {
      fetchEvidence(violation);
    }
  };

  const closeModal = () => {
    setModalOpen(false);
    setSelectedViolationId(null);
    setEvidenceType(null);
  };

  const handleStatusChange = (violationId, newStatus) => {
    setPendingStatusChanges(prev => ({
      ...prev,
      [violationId]: newStatus
    }));
  };

  const saveStatusChange = async (violationId) => {
    const newStatus = pendingStatusChanges[violationId];
    if (!newStatus) return;

    try {
      const violation = violations.find(v => v.violation_id === violationId);
      if (!violation) throw new Error('Violation not found');

      const payload = {
        violation: {
          ...violation,
          processed_time: violation.processed_time ? new Date(violation.processed_time).toISOString() : null
        },
        new_status: newStatus
      };

      const updatedViolation = await updateViolationStatus(payload);

      setViolations(violations.map(v => 
        v.violation_id === violationId 
          ? updatedViolation 
          : v
      ));
      setPendingStatusChanges(prev => {
        const newChanges = { ...prev };
        delete newChanges[violationId];
        return newChanges;
      });
    } catch (error) {
      console.error('Error updating violation status:', error);
      setError(error.message || 'Không thể cập nhật trạng thái vi phạm. Vui lòng thử lại sau.');
    }
  };

  const sortViolations = (key, direction) => {
    setSortConfig({ key, direction });
    const sorted = [...violations].sort((a, b) => {
      if (key === 'violation_time') {
        const dateA = new Date(a.violation_time);
        const dateB = new Date(b.violation_time);
        return direction === 'asc' ? dateA - dateB : dateB - dateA;
      } else if (key === 'status') {
        const statusA = statusOptions.find(s => s.value === a.status)?.label || a.status;
        const statusB = statusOptions.find(s => s.value === b.status)?.label || b.status;
        return direction === 'asc' 
          ? statusA.localeCompare(statusB, 'vi-VN') 
          : statusB.localeCompare(statusA, 'vi-VN');
      }
      return 0;
    });
    setViolations(sorted);
  };

  const indexOfLastViolation = currentPage * violationsPerPage;
  const indexOfFirstViolation = indexOfLastViolation - violationsPerPage;
  const currentViolations = violations.slice(indexOfFirstViolation, indexOfLastViolation);
  const totalPages = Math.ceil(violations.length / violationsPerPage);

  const handlePageChange = (pageNumber) => {
    setCurrentPage(pageNumber);
  };

  const handlePreviousPage = () => {
    if (currentPage > 1) {
      setCurrentPage(currentPage - 1);
    }
  };

  const handleNextPage = () => {
    if (currentPage < totalPages) {
      setCurrentPage(currentPage + 1);
    }
  };

  return (
    <div className="violations-query-container">
      <h2>Tra cứu vi phạm</h2>
      
      <form onSubmit={handleQuerySubmit} className="query-form">
        <div className="query-type-selector">
          <div 
            className={`query-tab ${queryType === 'date' ? 'active' : ''}`}
            onClick={() => setQueryType('date')}
          >
            Theo ngày
          </div>
          <div 
            className={`query-tab ${queryType === 'status' ? 'active' : ''}`}
            onClick={() => setQueryType('status')}
          >
            Theo trạng thái
          </div>
        </div>
        
        <div className="query-inputs">
          {queryType === 'date' ? (
            <div className="form-group">
              <label htmlFor="date-query">Ngày vi phạm:</label>
              <input 
                type="date" 
                id="date-query"
                value={dateQuery}
                onChange={(e) => setDateQuery(e.target.value)}
                required
              />
            </div>
          ) : (
            <div className="form-group">
              <label htmlFor="status-query">Trạng thái:</label>
              <select
                id="status-query"
                value={statusQuery}
                onChange={(e) => setStatusQuery(e.target.value)}
                required
              >
                {statusOptions.map(option => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </div>
          )}
          
          <button type="submit" className="query-button" disabled={loading}>
            {loading ? 'Đang truy vấn...' : 'Tra cứu'}
          </button>
        </div>
      </form>
      
      {error && <div className="error-message">{error}</div>}
      
      <div className="violations-results">
        {loading ? (
          <div className="loading">Đang tải dữ liệu vi phạm...</div>
        ) : violations.length > 0 ? (
          <>
            <div className="sort-controls">
              <button 
                className={`sort-btn ${sortConfig.key === 'violation_time' && sortConfig.direction === 'asc' ? 'active' : ''}`}
                onClick={() => sortViolations('violation_time', 'asc')}
              >
                Thời gian tăng dần
              </button>
              <button 
                className={`sort-btn ${sortConfig.key === 'violation_time' && sortConfig.direction === 'desc' ? 'active' : ''}`}
                onClick={() => sortViolations('violation_time', 'desc')}
              >
                Thời gian giảm dần
              </button>
              <button 
                className={`sort-btn ${sortConfig.key === 'status' && sortConfig.direction === 'asc' ? 'active' : ''}`}
                onClick={() => sortViolations('status', 'asc')}
              >
                Trạng thái A-Z
              </button>
              <button 
                className={`sort-btn ${sortConfig.key === 'status' && sortConfig.direction === 'desc' ? 'active' : ''}`}
                onClick={() => sortViolations('status', 'desc')}
              >
                Trạng thái Z-A
              </button>
            </div>
            <h3>Kết quả truy vấn ({violations.length}: vi phạm)</h3>
            <div className="violations-list">
              {currentViolations.map(violation => (
                <div 
                  key={violation.violation_id} 
                  className="violation-card" 
                  data-type={violation.violation_type}
                >
                  <div className="violation-header">
                    <div className="violation-basic-info">
                      <h4>{violationTypes[violation.violation_type] || violation.violation_type}</h4>
                      <div className="violation-metadata">
                        <span>Camera: {violation.camera_id}</span>
                        <span>Thời gian: {new Date(violation.violation_time).toLocaleString('vi-VN')}</span>
                        <span className={`violation-status status-${violation.status}`}>
                          {statusOptions.find(s => s.value === violation.status)?.label || violation.status}
                        </span>
                      </div>
                    </div>
                    <button 
                      className="toggle-details-btn"
                      onClick={() => toggleDetails(violation.violation_id)}
                    >
                      {showDetails[violation.violation_id] ? 'Thu gọn' : 'Chi tiết'}
                    </button>
                  </div>
                  
                  {showDetails[violation.violation_id] && (
                    <div className="violation-details">
                      <div className="violation-info">
                        <p><strong>ID:</strong> {violation.violation_id}</p>
                        <p><strong>Biển số xe:</strong> {violation.license_plate || 'Chưa xác định'}</p>
                        <p><strong>Thời gian xử lý:</strong> {violation.processed_time ? new Date(violation.processed_time).toLocaleString('vi-VN') : 'Chưa xử lý'}</p>
                        
                        <div className="violation-actions">
                          <div className="status-selector">
                            <label htmlFor={`status-${violation.violation_id}`}>Cập nhật trạng thái:</label>
                            <select 
                              id={`status-${violation.violation_id}`}
                              value={pendingStatusChanges[violation.violation_id] || violation.status}
                              onChange={(e) => handleStatusChange(violation.violation_id, e.target.value)}
                            >
                              {statusOptions.map(option => (
                                <option key={option.value} value={option.value}>
                                  {option.label}
                                </option>
                              ))}
                            </select>
                            <button 
                              className="save-status-btn"
                              onClick={() => saveStatusChange(violation.violation_id)}
                              disabled={!pendingStatusChanges[violation.violation_id]}
                            >
                              Lưu
                            </button>
                          </div>
                          <button className="export-btn">Xuất báo cáo</button>
                          <div className="evidence-buttons">
                            <button 
                              className="evidence-btn"
                              onClick={() => openEvidenceModal(violation.violation_id, 'image')}
                            >
                              Xem ảnh
                            </button>
                            <button 
                              className="evidence-btn"
                              onClick={() => openEvidenceModal(violation.violation_id, 'video')}
                            >
                              Xem video
                            </button>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>

            {totalPages > 1 && (
              <div className="pagination">
                <button
                  className="pagination-btn"
                  onClick={handlePreviousPage}
                  disabled={currentPage === 1}
                >
                  Trước
                </button>
                {Array.from({ length: totalPages }, (_, index) => index + 1).map(page => (
                  <button
                    key={page}
                    className={`pagination-btn ${currentPage === page ? 'active' : ''}`}
                    onClick={() => handlePageChange(page)}
                  >
                    {page}
                  </button>
                ))}
                <button
                  className="pagination-btn"
                  onClick={handleNextPage}
                  disabled={currentPage === totalPages}
                >
                  Sau
                </button>
              </div>
            )}
          </>
        ) : !loading && (
          <div className="no-results">Không tìm thấy vi phạm nào.</div>
        )}
      </div>

      {modalOpen && (
        <div className="evidence-modal">
          <div className="modal-content">
            <button className="close-modal-btn" onClick={closeModal}>Đóng</button>
            <div className="modal-body">
              {evidence[selectedViolationId] ? (
                evidenceType === 'image' ? (
                  <div className="image-container">
                    <div className="image-wrapper">
                      <h4>Ảnh sau vi phạm</h4>
                      <img 
                        src={evidence[selectedViolationId].imageUrl} 
                        alt={`Vi phạm ${selectedViolationId} - Sau`} 
                        className="modal-image"
                      />
                    </div>
                    <div className="image-wrapper">
                      <h4>Ảnh trước vi phạm</h4>
                      <img 
                        src={evidence[selectedViolationId].imageUrlBefore} 
                        alt={`Vi phạm ${selectedViolationId} - Trước`} 
                        className="modal-image"
                      />
                    </div>
                    {evidence[selectedViolationId].imageUrlPlate && (
                      <div className="image-wrapper">
                        <h4>Ảnh biển số</h4>
                        <img 
                          src={evidence[selectedViolationId].imageUrlPlate} 
                          alt={`Vi phạm ${selectedViolationId} - Biển số`} 
                          className="modal-image"
                        />
                      </div>
                    )}
                  </div>
                ) : (
                  <video 
                    controls 
                    src={evidence[selectedViolationId].videoUrl} 
                    className="modal-video"
                  >
                    Trình duyệt của bạn không hỗ trợ video.
                  </video>
                )
              ) : (
                <div className="loading-evidence">Đang tải {evidenceType === 'image' ? 'hình ảnh' : 'video'}...</div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ViolationsQuery;