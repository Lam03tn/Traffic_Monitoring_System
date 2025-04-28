import React, { useState, useEffect } from 'react';
import { fetchViolationsByDate, fetchViolationsByStatus } from '../services/apiServices';
import '../css/ViolationsQuery.css';

const ViolationsQuery = () => {
  const [queryType, setQueryType] = useState('date');
  const [dateQuery, setDateQuery] = useState(new Date().toISOString().split('T')[0]);
  const [statusQuery, setStatusQuery] = useState('pending');
  const [violations, setViolations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showDetails, setShowDetails] = useState({});

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

  const toggleDetails = (violationId) => {
    setShowDetails(prev => ({
      ...prev,
      [violationId]: !prev[violationId]
    }));
  };

  const handleStatusChange = async (violationId, newStatus) => {
    try {
      await updateViolationStatus(violationId, newStatus);
      
      // Update local state
      setViolations(violations.map(violation => 
        violation.id === violationId 
          ? { ...violation, status: newStatus } 
          : violation
      ));
    } catch (error) {
      console.error('Error updating violation status:', error);
      setError('Không thể cập nhật trạng thái vi phạm. Vui lòng thử lại sau.');
    }
  };

  // Mock function - replace with actual API call
  const updateViolationStatus = async (violationId, newStatus) => {
    // In a real implementation, you would call your API here
    console.log(`Updating violation ${violationId} to status ${newStatus}`);
    return Promise.resolve();
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
            <h3>Kết quả truy vấn ({violations.length} vi phạm)</h3>
            <div className="violations-list">
              {violations.map(violation => (
                <div key={violation.id} className="violation-card">
                  <div className="violation-header">
                    <div className="violation-basic-info">
                      <h4>{violationTypes[violation.violation_type] || violation.violation_type}</h4>
                      <div className="violation-metadata">
                        <span>Camera: {violation.camera_name}</span>
                        <span>Thời gian: {new Date(violation.timestamp).toLocaleString('vi-VN')}</span>
                        <span className={`violation-status status-${violation.status}`}>
                          {statusOptions.find(s => s.value === violation.status)?.label || violation.status}
                        </span>
                      </div>
                    </div>
                    <button 
                      className="toggle-details-btn"
                      onClick={() => toggleDetails(violation.id)}
                    >
                      {showDetails[violation.id] ? 'Thu gọn' : 'Chi tiết'}
                    </button>
                  </div>
                  
                  {showDetails[violation.id] && (
                    <div className="violation-details">
                      <div className="violation-image">
                        {violation.image_url ? (
                          <img src={violation.image_url} alt={`Vi phạm ${violation.id}`} />
                        ) : (
                          <div className="no-image">Không có hình ảnh</div>
                        )}
                      </div>
                      
                      <div className="violation-info">
                        <p><strong>ID:</strong> {violation.id}</p>
                        <p><strong>Vị trí:</strong> {violation.location}</p>
                        <p><strong>Biển số xe:</strong> {violation.license_plate || 'Chưa xác định'}</p>
                        <p><strong>Mô tả:</strong> {violation.description || 'Không có mô tả'}</p>
                        
                        <div className="violation-actions">
                          <div className="status-selector">
                            <label htmlFor={`status-${violation.id}`}>Cập nhật trạng thái:</label>
                            <select 
                              id={`status-${violation.id}`}
                              value={violation.status}
                              onChange={(e) => handleStatusChange(violation.id, e.target.value)}
                            >
                              {statusOptions.map(option => (
                                <option key={option.value} value={option.value}>
                                  {option.label}
                                </option>
                              ))}
                            </select>
                          </div>
                          
                          <button className="export-btn">Xuất báo cáo</button>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </>
        ) : !loading && (
          <div className="no-results">Không tìm thấy vi phạm nào.</div>
        )}
      </div>
    </div>
  );
};

export default ViolationsQuery;