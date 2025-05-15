import api, { transformRegionsToBackendSchema, transformBackendToFrontendRegions } from './apiHelpers';

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

/**
 * Saves camera violation configuration to the backend
 * @param {Object} configData - The configuration data including camera ID and violation parameters
 * @returns {Promise<Object>} - A promise that resolves to the saved configuration
 */
export const saveCameraViolationConfig = async (configData) => {
  try {
    const violationConfig = transformRegionsToBackendSchema(
      configData.parameters, 
      configData.violation_type
    );
    const payload = {
      cam_id: configData.cam_id,
      violation_type: configData.violation_type,
      violation_config: violationConfig,
      metadata: {
        description: configData.description,
        thumbnail: configData.thumbnail
      }
    };

    if (configData.isUpdate) {
      const response = await api.put(`/violation-camera-config/${configData.cam_id}`, payload);
      return response.data;
    } else {
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
        if (response.data) {
          activeViolations.push(violationType);
        }
      } catch (error) {
        if (error.response?.status !== 404) {
          throw new Error(error.response?.data?.message || `Failed to fetch ${violationType} violation`);
        }
      }
    }

    return activeViolations;
  } catch (error) {
    throw new Error(error.message || 'Failed to fetch active violations');
  }
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
    
    const frontendData = {
      ...response.data,
      regions: transformBackendToFrontendRegions(response.data)
    };
    
    return frontendData;
  } catch (error) {
    if (error.response && error.response.status === 404) {
      return null;
    }
    throw new Error(error.response?.data?.message || 'Failed to fetch violation configuration');
  }
};