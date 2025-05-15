import api from './apiHelpers';

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
export const updateViolationStatus = async (payload) => {
  try {
    const response = await api.post(`/query/violations/update-status`, payload);
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.message || 'Failed to update violation status');
  }
};

/**
 * Fetches violation video evidence
 * @param {string} cameraId - ID of the camera
 * @param {string} timestamp - Timestamp of the violation (YYYYMMDDHHMMSS)
 * @returns {Promise<Blob>} - A promise that resolves to the video blob
 */
export const fetchViolationVideo = async (violation_type, camera_id, timestamp) => {
  try {
    const response = await api.get(`/query/evidence/video/${violation_type}/${camera_id}/${timestamp}`, {
      responseType: 'blob'
    });
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.message || 'Failed to fetch violation video');
  }
};
/**
 * Fetches violation image evidence
 * @param {string} cameraId - ID of the camera
 * @param {string} timestamp - Timestamp of the violation (YYYYMMDDHHMMSS)
 * @returns {Promise<Blob>} - A promise that resolves to the image blob
 */
export const fetchViolationImage = async (violation_type, camera_id, timestamp) => {
  try {
    const response = await api.get(`/query/evidence/image/${violation_type}/${camera_id}/${timestamp}`, {
      responseType: 'blob'
    });
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.message || 'Failed to fetch violation image');
  }
};