import React, { useState, useRef, useEffect } from 'react';
import { saveCameraViolationConfig, getViolationConfig } from '../services/cameraConfigService';

const AddViolationModal = ({ camera, onClose, videoDimensions, onRefreshViolations }) => {
  const [step, setStep] = useState(1);
  const [violationType, setViolationType] = useState('');
  const [description, setDescription] = useState('');
  const [timestamp, setTimestamp] = useState(new Date().toISOString().slice(0, 16));
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState(null);
  const [capturedFrame, setCapturedFrame] = useState(null);
  const [regions, setRegions] = useState([]);
  const [activeParameter, setActiveParameter] = useState(null);
  const canvasRef = useRef(null);
  const frameRef = useRef(null);
  const currentPointsRef = useRef([]);
  const [frameAspectRatio, setFrameAspectRatio] = useState(16 / 9);
  const [isConfigUpdate, setIsConfigUpdate] = useState(false);
  const [saveSuccess, setSaveSuccess] = useState(false);

  // Violation categories with required parameters
  const violationTypes = {
    traffic_light: {
      label: 'Vượt đèn đỏ',
      parameters: [
        { name: 'traffic_light_area', label: 'Vùng đèn giao thông', color: '#ff0000', type: 'area', pointCount: 4 },
        { name: 'stop_line', label: 'Vạch dừng', color: '#ffff00', type: 'line', pointCount: 2 },
        { name: 'detection_zone', label: 'Vùng phát hiện (ROI)', color: '#00ff00', type: 'area', pointCount: 4 },
      ],
    },
    wrong_way: {
      label: 'Đi ngược chiều',
      parameters: [
        { name: 'direction_line', label: 'Vạch xác định hướng', color: '#0000ff', type: 'line', pointCount: 2 },
        { name: 'detection_zone', label: 'Vùng phát hiện (ROI)', color: '#00ff00', type: 'area', pointCount: 4 },
      ],
    },
    illegal_parking: {
      label: 'Đỗ xe trái phép',
      parameters: [
        { name: 'no_parking_zone', label: 'Vùng cấm đỗ xe (ROI)', color: '#ff00ff', type: 'area', pointCount: 4 },
      ],
    },
    speeding: {
      label: 'Vượt tốc độ',
      parameters: [
        { name: 'speed_measurement_start', label: 'Điểm bắt đầu đo', color: '#00ffff', type: 'line', pointCount: 2 },
        { name: 'speed_measurement_end', label: 'Điểm kết thúc đo', color: '#ff8800', type: 'line', pointCount: 2 },
      ],
    },
    other: {
      label: 'Vi phạm khác',
      parameters: [
        { name: 'custom_zone', label: 'Vùng tùy chọn (ROI)', color: '#00ff00', type: 'area', pointCount: 4 },
      ],
    },
  };

  // Fetch existing configuration when violation type changes
  useEffect(() => {
    if (violationType) {
      const fetchExistingConfig = async () => {
        try {
          const config = await getViolationConfig(camera.id, violationType);
          
          if (config && config.parameters) {
            setIsConfigUpdate(true);
            
            // Transform parameters to match our regions structure
            const existingRegions = config.parameters.map(param => {
              const parameterDef = violationTypes[violationType].parameters.find(
                p => p.name === param.type
              );
              
              return {
                type: param.type,
                label: parameterDef ? parameterDef.label : param.type,
                color: parameterDef ? parameterDef.color : '#00ff00',
                paramType: param.paramType,
                points: param.points
              };
            });
            
            setRegions(existingRegions);
            
            if (config.description) {
              setDescription(config.description);
            }
          } else {
            setIsConfigUpdate(false);
            setRegions([]);
          }
        } catch (error) {
          console.error('Error fetching configuration:', error);
          setIsConfigUpdate(false);
        }
      };
      
      fetchExistingConfig();
    }
  }, [violationType, camera.id]);

  // Kiểm tra trạng thái video và chụp khung hình
  const captureVideoFrame = async () => {
    const videoElement = document.querySelector('.video-player video');
    if (!videoElement) {
      setError('Không tìm thấy luồng video.');
      return;
    }
    if (!videoDimensions.width || !videoDimensions.height) {
      setError('Kích thước video không hợp lệ.');
      return;
    }

    // Đợi video sẵn sàng
    if (videoElement.readyState < 2) {
      try {
        await new Promise((resolve) => {
          videoElement.onloadedmetadata = resolve;
          setTimeout(resolve, 2000); // Timeout sau 2 giây
        });
      } catch (error) {
        console.error('Error waiting for video metadata:', error);
        setError('Không thể tải metadata video.');
        return;
      }
    }

    try {
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = videoDimensions.width;
      tempCanvas.height = videoDimensions.height;
      const ctx = tempCanvas.getContext('2d');
      ctx.drawImage(videoElement, 0, 0, tempCanvas.width, tempCanvas.height);
      const frameDataUrl = tempCanvas.toDataURL('image/jpeg');
      setCapturedFrame(frameDataUrl);
      setError(null);
    } catch (error) {
      console.error('Error capturing video frame:', error);
      setError('Không thể chụp khung hình từ video.');
    }
  };

  // Chụp khung hình khi chuyển sang bước 2
  useEffect(() => {
    if (step === 2 && !capturedFrame) {
      captureVideoFrame();
    }
  }, [step]);

  // Thiết lập canvas với kích thước thực tế
  useEffect(() => {
    if (capturedFrame && canvasRef.current && videoDimensions.width && videoDimensions.height) {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');

      const img = new Image();
      img.onload = () => {
        canvas.width = videoDimensions.width;
        canvas.height = videoDimensions.height;
        setFrameAspectRatio(videoDimensions.width / videoDimensions.height);
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        frameRef.current = img;
        drawExistingRegions();
      };
      img.onerror = () => {
        setError('Không thể tải khung hình đã chụp.');
      };
      img.src = capturedFrame;
    }
  }, [capturedFrame, videoDimensions]);

  // Cập nhật active parameter
  useEffect(() => {
    if (step === 2 && !activeParameter) {
      const firstParam = getNextParameterToDraw();
      if (firstParam) {
        setActiveParameter(firstParam);
        currentPointsRef.current = [];
      }
    }
  }, [step, regions, activeParameter]);

  const handleViolationTypeChange = (e) => {
    const type = e.target.value;
    setViolationType(type);
    // Regions will be set by the useEffect that fetches existing config
  };

  const proceedToDrawing = () => {
    if (!violationType) {
      setError('Vui lòng chọn loại vi phạm');
      return;
    }
    setError(null);
    setStep(2);
    setActiveParameter(null);
    currentPointsRef.current = [];
  };

  const handleCanvasClick = (e) => {
    if (!activeParameter) return;

    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();

    const scaleX = videoDimensions.width / rect.width;
    const scaleY = videoDimensions.height / rect.height;

    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;

    if (x >= 0 && x <= canvas.width && y >= 0 && y <= canvas.height) {
      const newPoints = [...currentPointsRef.current, { x, y }];
      currentPointsRef.current = newPoints;
      drawTempPoints();

      if (newPoints.length === activeParameter.pointCount) {
        finalizeRegion();
      }
    }
  };

  const finalizeRegion = () => {
    if (!activeParameter || currentPointsRef.current.length !== activeParameter.pointCount) return;

    let points = [...currentPointsRef.current];
    if (activeParameter.type === 'area' && points.length === 4) {
      points = sortPointsConvex(points);
    }

    const newRegion = {
      type: activeParameter.name,
      label: activeParameter.label,
      color: activeParameter.color,
      paramType: activeParameter.type,
      points,
    };

    const existingIndex = regions.findIndex((r) => r.type === activeParameter.name);
    let newRegions;
    if (existingIndex !== -1) {
      newRegions = [...regions];
      newRegions[existingIndex] = newRegion;
    } else {
      newRegions = [...regions, newRegion];
    }

    setRegions(newRegions);

    const nextParam = getNextParameterToDraw(newRegions);
    setActiveParameter(nextParam);
    currentPointsRef.current = [];

    setTimeout(() => {
      redrawCanvas();
    }, 0);
  };

  const sortPointsConvex = (points) => {
    const centroid = calculateCentroid(points);
    return points.sort((a, b) => {
      const angleA = Math.atan2(a.y - centroid.y, a.x - centroid.x);
      const angleB = Math.atan2(b.y - centroid.y, b.x - centroid.x);
      return angleA - angleB;
    });
  };

  const getNextParameterToDraw = (currentRegions = regions) => {
    if (!violationType || !violationTypes[violationType]) return null;

    const parameters = violationTypes[violationType].parameters;
    if (!parameters || parameters.length === 0) return null;

    const drawnTypes = currentRegions.map((r) => r.type);

    for (const param of parameters) {
      if (!drawnTypes.includes(param.name)) {
        return param;
      }
    }

    return null;
  };

  const drawTempPoints = () => {
    if (!canvasRef.current || !frameRef.current || !activeParameter) return;

    redrawCanvas();
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    const points = currentPointsRef.current;
    if (points.length > 0) {
      ctx.fillStyle = activeParameter.color;
      points.forEach((point, index) => {
        ctx.beginPath();
        ctx.arc(point.x, point.y, 5, 0, 2 * Math.PI);
        ctx.fill();
        ctx.fillText((index + 1).toString(), point.x + 8, point.y - 8);
      });

      ctx.beginPath();
      ctx.moveTo(points[0].x, points[0].y);
      for (let i = 1; i < points.length; i++) {
        ctx.lineTo(points[i].x, points[i].y);
      }

      if (activeParameter.type === 'area' && points.length === activeParameter.pointCount - 1) {
        ctx.lineTo(points[0].x, points[0].y);
      }

      ctx.strokeStyle = activeParameter.color;
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  };

  const redrawCanvas = () => {
    if (!canvasRef.current || !frameRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(frameRef.current, 0, 0);

    drawExistingRegions();
  };

  const drawExistingRegions = () => {
    if (!canvasRef.current) return;

    const ctx = canvasRef.current.getContext('2d');

    regions.forEach((region) => {
      drawRegion(ctx, region);
    });
  };

  const drawRegion = (ctx, region) => {
    if (!region.points || region.points.length < 2) return;

    ctx.beginPath();
    ctx.moveTo(region.points[0].x, region.points[0].y);

    for (let i = 1; i < region.points.length; i++) {
      ctx.lineTo(region.points[i].x, region.points[i].y);
    }

    if (region.paramType === 'area') {
      ctx.closePath();
      ctx.fillStyle = region.color + '33';
      ctx.fill();
    }

    ctx.strokeStyle = region.color;
    ctx.lineWidth = 3;
    ctx.stroke();

    region.points.forEach((point, index) => {
      ctx.beginPath();
      ctx.arc(point.x, point.y, 5, 0, 2 * Math.PI);
      ctx.fillStyle = region.color;
      ctx.fill();
      ctx.fillText((index + 1).toString(), point.x + 8, point.y - 8);
    });

    const labelPoint =
      region.paramType === 'area'
        ? calculateCentroid(region.points)
        : calculateMidpoint(region.points[0], region.points[region.points.length - 1]);

    ctx.fillStyle = region.color;
    ctx.font = '14px Arial';
    ctx.fillText(region.label, labelPoint.x, labelPoint.y);
  };

  const calculateCentroid = (points) => {
    let sumX = 0;
    let sumY = 0;

    points.forEach((point) => {
      sumX += point.x;
      sumY += point.y;
    });

    return {
      x: sumX / points.length,
      y: sumY / points.length,
    };
  };

  const calculateMidpoint = (point1, point2) => {
    return {
      x: (point1.x + point2.x) / 2,
      y: (point1.y + point2.y) / 2,
    };
  };

  const resetActiveParameter = () => {
    currentPointsRef.current = [];
    redrawCanvas();
  };

  const removeRegion = (regionType) => {
    const newRegions = regions.filter((r) => r.type !== regionType);
    setRegions(newRegions);

    if (activeParameter && activeParameter.name === regionType) {
      setActiveParameter({ ...activeParameter });
      currentPointsRef.current = [];
    } else if (!activeParameter) {
      const param = violationTypes[violationType].parameters.find((p) => p.name === regionType);
      if (param) {
        setActiveParameter(param);
        currentPointsRef.current = [];
      }
    }

    redrawCanvas();
  };

  const selectParameter = (param) => {
    setActiveParameter(param);
    currentPointsRef.current = [];
    redrawCanvas();
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    const requiredParams = violationTypes[violationType].parameters;
    const missingParams = requiredParams.filter((param) => !regions.some((r) => r.type === param.name));

    if (missingParams.length > 0) {
      setError(`Vui lòng thiết lập tất cả các thông số: ${missingParams.map((p) => p.label).join(', ')}`);
      return;
    }

    setIsSubmitting(true);
    setError(null);

    try {
      const configData = {
        cam_id: camera.id,
        violation_type: violationType,
        violation_type_name: violationTypes[violationType].label,
        description,
        parameters: regions.map((region) => ({
          type: region.type,
          paramType: region.paramType,
          points: region.points,
        })),
        thumbnail: capturedFrame,
        isUpdate: isConfigUpdate
      };

      console.log('Saving violation configuration:', configData);
      
      // Send to server
      const result = await saveCameraViolationConfig(configData);
      console.log('Configuration saved successfully:', result);
      
      // Refresh violations after saving
      if (onRefreshViolations.current) {
        await onRefreshViolations.current();
      }
      
      setSaveSuccess(true);
      setTimeout(() => {
        setSaveSuccess(false);
        onClose();
      }, 1500);
    } catch (error) {
      console.error('Error saving configuration:', error);
      setError('Không thể lưu cấu hình. Vui lòng thử lại sau.');
      setIsSubmitting(false);
    }
  };

  const handleBack = () => {
    setStep(1);
    // Don't reset regions as we might want to keep the existing configuration
    setCapturedFrame(null);
    currentPointsRef.current = [];
    setActiveParameter(null);
  };

  return (
    <div className="modal-overlay">
      <div className="violation-modal">
        <div className="modal-header">
          <h2>
            {step === 1 
              ? (isConfigUpdate ? 'Chỉnh sửa cấu hình vi phạm' : 'Thêm cấu hình vi phạm mới') 
              : 'Thiết lập thông số vi phạm'}
          </h2>
          <button className="close-btn" onClick={onClose} aria-label="Đóng">
            ×
          </button>
        </div>

        <div className="modal-body">
          {error && <div className="error-message">{error}</div>}
          {saveSuccess && (
            <div className="success-message" style={{ backgroundColor: '#d4edda', color: '#155724', padding: '0.75rem', borderRadius: '4px', marginBottom: '1rem' }}>
              Cấu hình đã được lưu thành công!
            </div>
          )}

          {step === 1 ? (
            <>
              <div className="camera-details">
                <p>
                  <strong>Camera:</strong> {camera.name}
                </p>
                <p>
                  <strong>Vị trí:</strong> {camera.location}
                </p>
              </div>

              <form>
                <div className="form-group">
                  <label htmlFor="violation-type">Loại vi phạm:</label>
                  <select
                    id="violation-type"
                    value={violationType}
                    onChange={handleViolationTypeChange}
                    required
                  >
                    <option value="">Chọn loại vi phạm</option>
                    {Object.entries(violationTypes).map(([key, type]) => (
                      <option key={key} value={key}>
                        {type.label}
                      </option>
                    ))}
                  </select>
                </div>

                <div className="form-group">
                  <label htmlFor="violation-description">Mô tả:</label>
                  <textarea
                    id="violation-description"
                    value={description}
                    onChange={(e) => setDescription(e.target.value)}
                    rows="4"
                    placeholder="Mô tả chi tiết về vi phạm..."
                  />
                </div>

                <div className="form-actions">
                  <button type="button" onClick={onClose}>
                    Hủy
                  </button>
                  <button type="button" onClick={proceedToDrawing} disabled={!violationType}>
                    Tiếp tục
                  </button>
                </div>
              </form>
            </>
          ) : (
            <>
              <div className="drawing-instructions">
                <p>
                  <strong>Loại vi phạm:</strong> {violationType && violationTypes[violationType].label}
                </p>

                {violationType && (
                  <div className="parameters-list">
                    <p>
                      <strong>Các thông số cần thiết lập:</strong>
                    </p>
                    <ul>
                      {violationTypes[violationType].parameters.map((param, index) => {
                        const isDrawn = regions.some((r) => r.type === param.name);
                        const isActive = activeParameter && activeParameter.name === param.name;

                        return (
                          <li
                            key={index}
                            className={`parameter-item ${isDrawn ? 'drawn' : ''} ${isActive ? 'active' : ''}`}
                            onClick={() => selectParameter(param)}
                          >
                            <span
                              className="parameter-color"
                              style={{ backgroundColor: param.color }}
                            ></span>
                            <span className="parameter-name">
                              {param.label} {param.type === 'area' ? ' (4 điểm)' : ' (2 điểm)'}
                            </span>
                            {isDrawn && (
                              <button
                                type="button"
                                className="remove-region-btn"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  removeRegion(param.name);
                                }}
                              >
                                ×
                              </button>
                            )}
                          </li>
                        );
                      })}
                    </ul>

                    {activeParameter && (
                      <div className="current-drawing-instruction">
                        <p>
                          <strong>Đang vẽ:</strong> {activeParameter.label}
                          {activeParameter.type === 'area'
                            ? ` (4 điểm, đã chọn: ${currentPointsRef.current.length}/4)`
                            : ` (2 điểm, đã chọn: ${currentPointsRef.current.length}/2)`}
                        </p>
                        <button
                          type="button"
                          className="reset-points-btn"
                          onClick={resetActiveParameter}
                          disabled={currentPointsRef.current.length === 0}
                        >
                          Vẽ lại điểm
                        </button>
                      </div>
                    )}

                    {!activeParameter &&
                      regions.length === violationTypes[violationType].parameters.length && (
                        <div className="all-parameters-drawn">
                          <p>
                            Tất cả các thông số đã được thiết lập. Bạn có thể chỉnh sửa bằng cách chọn
                            thông số cần thay đổi.
                          </p>
                        </div>
                      )}
                  </div>
                )}

                <div className="canvas-container">
                  {capturedFrame ? (
                    <canvas
                      ref={canvasRef}
                      onClick={handleCanvasClick}
                      style={{
                        maxWidth: '100%',
                        height: 'auto',
                        border: '1px solid #ddd',
                        cursor: activeParameter ? 'crosshair' : 'default',
                      }}
                    />
                  ) : (
                    <div className="loading">Đang tải khung hình...</div>
                  )}
                </div>

                <div className="form-actions drawing-actions">
                  <button type="button" onClick={handleBack}>
                    Quay lại
                  </button>
                  <button
                    type="button"
                    onClick={handleSubmit}
                    disabled={
                      isSubmitting ||
                      !violationType ||
                      regions.length < violationTypes[violationType]?.parameters.length
                    }
                  >
                    {isSubmitting ? 'Đang lưu...' : (isConfigUpdate ? 'Cập nhật cấu hình' : 'Lưu cấu hình')}
                  </button>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default AddViolationModal;