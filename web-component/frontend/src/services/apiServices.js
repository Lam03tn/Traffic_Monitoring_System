// services/apiService.js
import axios from 'axios';

const api = axios.create({
  baseURL: process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000',
  headers: {
    'Content-Type': 'application/json',
  },
});

/**
 * Retrieves all cameras from the API
 * @returns {Promise<Array>} - A promise that resolves to an array of camera objects
 */
export const fetchCameras = async () => {
  try {
    const response = await api.get('/camera-config/all-cameras');
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.message || 'Failed to fetch cameras');
  }
};

// /**
//  * Submits a violation report to the API
//  * @param {Object} violationData - The violation data to submit
//  * @returns {Promise<Object>} - A promise that resolves to the response data
//  */
// export const submitViolation = async (violationData) => {
//   try {
//     const response = await api.post('/violations/report', violationData);
//     return response.data;
//   } catch (error) {
//     throw new Error(error.response?.data?.message || 'Failed to submit violation');
//   }
// };

/**
 * Transforms frontend regions data to backend schema format
 */
const transformRegionsToBackendSchema = (regions, violationType) => {
  const configs = [];
  
  // Always need ROI for all violation types
  const roi = regions.find(r => r.type === 'detection_zone');
  if (!roi) throw new Error('ROI (detection_zone) is required for all violation types');
  
  const roiConfig = {
    roi: {
      point1: roi.points[0],
      point2: roi.points[1],
      point3: roi.points[2],
      point4: roi.points[3],
    }
  };
  
  // Add specific parameters based on violation type
  switch(violationType) {
    case 'traffic_light':
      const trafficLightZone = regions.find(r => r.type === 'traffic_light_area');
      const stopLine = regions.find(r => r.type === 'stop_line');
      
      if (!trafficLightZone) throw new Error('Traffic light zone is required for red light violation');
      if (!stopLine) throw new Error('Stop line is required for red light violation');
      
      configs.push({
        ...roiConfig,
        traffic_light_zone: {
          point1: trafficLightZone.points[0],
          point2: trafficLightZone.points[1],
          point3: trafficLightZone.points[2],
          point4: trafficLightZone.points[3]
        },
        lane_marking: {
          start_point: stopLine.points[0],
          end_point: stopLine.points[1]
        }
      });
      break;
      
    case 'wrong_way':
      const directionLine = regions.find(r => r.type === 'direction_line');
      
      if (!directionLine) throw new Error('Direction line is required for wrong way violation');
      
      configs.push({
        ...roiConfig,
        lane_marking: {
          start_point: directionLine.points[0],
          end_point: directionLine.points[1]
        }
      });
      break;
      
    case 'illegal_parking':
      // Only needs ROI
      configs.push(roiConfig);
      break;
      
    case 'speeding':
      const startLine = regions.find(r => r.type === 'speed_measurement_start');
      const endLine = regions.find(r => r.type === 'speed_measurement_end');
      
      if (!startLine || !endLine) throw new Error('Both start and end lines are required for speeding violation');
      
      configs.push({
        ...roiConfig,
        lane_marking: {
          start_point: startLine.points[0],
          end_point: startLine.points[1]
        }
      }, {
        ...roiConfig,
        lane_marking: {
          start_point: endLine.points[0],
          end_point: endLine.points[1]
        }
      });
      break;
      
    default:
      // For other types, just use ROI
      configs.push(roiConfig);
  }
  
  return configs;
};

/**
 * Saves camera violation configuration to the backend
 * @param {Object} configData - The configuration data including camera ID and violation parameters
 * @returns {Promise<Object>} - A promise that resolves to the saved configuration
 */
export const saveCameraViolationConfig = async (configData) => {
  try {
    // Transform the frontend regions format to backend schema
    const violationConfig = transformRegionsToBackendSchema(
      configData.parameters, 
      configData.violation_type
    );
    const payload = {
      cam_id: configData.cam_id,
      violation_type: configData.violation_type,
      violation_config: violationConfig,
      // Include additional metadata if needed
      metadata: {
        description: configData.description,
        thumbnail: configData.thumbnail
      }
    };


    // If updating an existing config
    if (configData.isUpdate) {
      const response = await api.put(`/violation-camera-config/${configData.cam_id}`, payload);
      return response.data;
    } 
    // For new configuration
    else {
      const response = await api.post('/violation-camera-config/', payload);
      return response.data;
    }
  } catch (error) {
    throw new Error(error.response?.data?.message || 'Failed to save violation configuration');
  }
};

/**
 * Gets active violations for a specific camera by checking predefined violation types
 * @param {string} cameraId - The ID of the camera
 * @returns {Promise<Array>} - A promise that resolves to an array of active violations
 */
export const getCameraActiveViolations = async (cameraId) => {
  const violationTypes = [
    'wrong_way',
    'traffic_light',
    'speeding',
    'illegal_parking'
  ];

  try {
    const activeViolations = [];

    for (const violationType of violationTypes) {
      try {
        const response = await api.get(`/violation-camera-config/${cameraId}/${violationType}`);
        // console.log(`Response for ${violationType}:`, response.data); // Debug response thực tế
        if (response.data) {
          activeViolations.push(violationType);
        }
      } catch (error) {
        // Chỉ bỏ qua lỗi 404, các lỗi khác sẽ được ném ra ngoài
        if (error.response?.status !== 404) {
          throw new Error(error.response?.data?.message || `Failed to fetch ${violationType} violation`);
        }
        // Nếu là 404 thì tiếp tục vòng lặp mà không làm gì
      }
    }

    return activeViolations;
  } catch (error) {
    throw new Error(error.message || 'Failed to fetch active violations');
  }
};

// /**
//  * Gets all active violations across all cameras
//  * @returns {Promise<Array>} - A promise that resolves to an array of active violations
//  */
// export const getAllActiveViolations = async () => {
//   try {
//     const response = await api.get('/violations/active');
//     return response.data;
//   } catch (error) {
//     throw new Error(error.response?.data?.message || 'Failed to fetch active violations');
//   }
// };

/**
 * Transforms backend schema data to frontend regions format
 */
const transformBackendToFrontendRegions = (configData) => {
  const regions = [];
  
  // Get the first config (assuming all configs have the same ROI)
  const firstConfig = configData.violation_config[0];
  
  // Add ROI
  if (firstConfig.roi) {
    regions.push({
      type: 'detection_zone',
      label: 'Vùng phát hiện (ROI)',
      color: '#00ff00',
      paramType: 'area',
      points: [
        firstConfig.roi.point1,
        firstConfig.roi.point2,
        firstConfig.roi.point3,
        firstConfig.roi.point4
      ]
    });
  }
  
  // Add specific parameters based on violation type
  switch(configData.violation_type) {
    case 'traffic_light':
      if (firstConfig.traffic_light_zone) {
        regions.push({
          type: 'traffic_light_area',
          label: 'Vùng đèn giao thông',
          color: '#ff0000',
          paramType: 'area',
          points: [
            firstConfig.traffic_light_zone.point1,
            firstConfig.traffic_light_zone.point2,
            firstConfig.traffic_light_zone.point3,
            firstConfig.traffic_light_zone.point4
          ]
        });
      }
      
      if (firstConfig.lane_marking) {
        regions.push({
          type: 'stop_line',
          label: 'Vạch dừng',
          color: '#ffff00',
          paramType: 'line',
          points: [
            firstConfig.lane_marking.start_point,
            firstConfig.lane_marking.end_point
          ]
        });
      }
      break;
      
    case 'wrong_way':
      if (firstConfig.lane_marking) {
        regions.push({
          type: 'direction_line',
          label: 'Vạch xác định hướng',
          color: '#0000ff',
          paramType: 'line',
          points: [
            firstConfig.lane_marking.start_point,
            firstConfig.lane_marking.end_point
          ]
        });
      }
      break;
      
    case 'speeding':
      // Speeding has two configs - one for start line, one for end line
      if (configData.violation_config.length >= 2) {
        const startConfig = configData.violation_config[0];
        const endConfig = configData.violation_config[1];
        
        if (startConfig.lane_marking) {
          regions.push({
            type: 'speed_measurement_start',
            label: 'Điểm bắt đầu đo',
            color: '#00ffff',
            paramType: 'line',
            points: [
              startConfig.lane_marking.start_point,
              startConfig.lane_marking.end_point
            ]
          });
        }
        
        if (endConfig.lane_marking) {
          regions.push({
            type: 'speed_measurement_end',
            label: 'Điểm kết thúc đo',
            color: '#ff8800',
            paramType: 'line',
            points: [
              endConfig.lane_marking.start_point,
              endConfig.lane_marking.end_point
            ]
          });
        }
      }
      break;
  }
  
  return regions;
};

/**
 * Gets specific violation configuration for camera and violation type
 * @param {string} cameraId - The ID of the camera
 * @param {string} violationType - The type of violation
 * @returns {Promise<Object>} - A promise that resolves to the violation configuration
 */
export const getViolationConfig = async (cameraId, violationType) => {
  try {
    const response = await api.get(`/violation-camera-config/${cameraId}/${violationType}`);
    
    // Transform backend format to frontend format
    const frontendData = {
      ...response.data,
      regions: transformBackendToFrontendRegions(response.data)
    };
    
    return frontendData;
  } catch (error) {
    if (error.response && error.response.status === 404) {
      return null; // No configuration exists yet
    }
    throw new Error(error.response?.data?.message || 'Failed to fetch violation configuration');
  }
};

/**
 * Fetches violations by date
 * @param {string} date - Date in YYYY-MM-DD format
 * @returns {Promise<Array>} - A promise that resolves to an array of violation objects
 */
export const fetchViolationsByDate = async (date) => {
  try {
    const response = await api.get(`/query/violations/by-date/${date}`);
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.message || 'Failed to fetch violations by date');
  }
};

/**
 * Fetches violations by status
 * @param {string} status - Status to filter by (pending, processed, etc.)
 * @returns {Promise<Array>} - A promise that resolves to an array of violation objects
 */
export const fetchViolationsByStatus = async (status) => {
  try {
    const response = await api.get(`/query/violations/by-status/${status}`);
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.message || 'Failed to fetch violations by status');
  }
};

/**
 * Updates a violation's status
 * @param {string} violationId - ID of the violation to update
 * @param {string} newStatus - New status value
 * @returns {Promise<Object>} - A promise that resolves to the updated violation object
 */
export const updateViolationStatus = async (violationId, newStatus) => {
  try {
    const response = await api.put(`/violations/${violationId}/status`, { status: newStatus });
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.message || 'Failed to update violation status');
  }
};

// Export axios instance for other uses if needed
export default api;