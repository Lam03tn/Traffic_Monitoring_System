import axios from 'axios';

const api = axios.create({
  baseURL: process.env.REACT_APP_API_BASE_URL || 'http://127.0.0.1:8000',
  headers: {
    'Content-Type': 'application/json',
  },
});

/**
 * Transforms frontend regions data to backend schema format
 */
const transformRegionsToBackendSchema = (regions, violationType) => {
  const configs = [];
  
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
      configs.push(roiConfig);
  }
  
  return configs;
};

/**
 * Transforms backend schema data to frontend regions format
 */
const transformBackendToFrontendRegions = (configData) => {
  const regions = [];
  
  const firstConfig = configData.violation_config[0];
  
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

export default api; // Export api làm default
export { transformRegionsToBackendSchema, transformBackendToFrontendRegions }; // Export named