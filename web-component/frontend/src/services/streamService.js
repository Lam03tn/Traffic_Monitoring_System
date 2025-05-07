// In your React component
import { useEffect, useState, useRef } from 'react';

function VideoStream({ cameraId }) {
  const [status, setStatus] = useState('Connecting...');
  const wsRef = useRef(null);
  const imgRef = useRef(null);

  useEffect(() => {
    // Connect to WebSocket for the specific camera
    const ws = new WebSocket(`ws://your-api-host/ws/stream/${cameraId}`);
    
    ws.onopen = () => {
      setStatus('Connected');
    };
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (imgRef.current && data.frame) {
        imgRef.current.src = `data:image/jpeg;base64,${data.frame}`;
      }
    };
    
    ws.onerror = (error) => {
      setStatus('Error: ' + error.message);
    };
    
    ws.onclose = () => {
      setStatus('Disconnected');
    };
    
    wsRef.current = ws;
    
    // Clean up on unmount
    return () => {
      if (ws) {
        ws.close();
      }
    };
  }, [cameraId]);

  return (
    <div>
      <div>{status}</div>
      <img ref={imgRef} alt={`Camera ${cameraId} stream`} />
    </div>
  );
}