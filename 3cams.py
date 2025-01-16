import cv2
import numpy as np
from datetime import datetime
import os
from typing import List, Dict, Optional
import threading
import time

class MultiCameraCapture:
    def __init__(self, camera_ids: List[int] = [0,3,2]):
        """
        Initialize multi-camera capture system
        
        Args:
            camera_ids: List of camera device IDs to use
        """
        self.camera_ids = camera_ids
        self.cameras: Dict[int, cv2.VideoCapture] = {}
        self.frames: Dict[int, Optional[np.ndarray]] = {}
        self.running = True
        
        # Reduce frame size
        self.frame_width = 320  # Smaller frame width
        self.frame_height = 240  # Smaller frame height
        
        # Create output directory
        self.output_dir = "captured_images"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize cameras
        self._init_cameras()
        
        # Create capture button area
        self.button_height = 40  # Smaller button
        self.button_pressed = False
        
        # Start capture threads
        self.capture_threads = []
        self._start_capture_threads()

    def _init_cameras(self):
        """Initialize all cameras with retry mechanism"""
        for cam_id in self.camera_ids:
            retry_count = 0
            max_retries = 3
            
            while retry_count < max_retries:
                try:
                    cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)  # Add CAP_DSHOW
                    if not cap.isOpened():
                        print(f"Warning: Unable to open camera {cam_id}, attempt {retry_count + 1}")
                        retry_count += 1
                        time.sleep(1)  # Wait before retrying
                        continue
                    
                    # Set resolution for each camera
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
                    
                    # Test grab a frame
                    ret = cap.grab()
                    if not ret:
                        print(f"Warning: Unable to grab frame from camera {cam_id}, attempt {retry_count + 1}")
                        cap.release()
                        retry_count += 1
                        time.sleep(1)
                        continue
                    
                    self.cameras[cam_id] = cap
                    self.frames[cam_id] = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
                    break
                    
                except Exception as e:
                    print(f"Error initializing camera {cam_id}: {str(e)}")
                    retry_count += 1
                    time.sleep(1)
            
            if retry_count == max_retries:
                print(f"Failed to initialize camera {cam_id} after {max_retries} attempts")

        if not self.cameras:
            raise RuntimeError("No cameras could be initialized")

    def _capture_thread(self, camera_id: int):
        """Thread function for capturing frames from a camera"""
        while self.running:
            try:
                if camera_id in self.cameras:
                    ret, frame = self.cameras[camera_id].read()
                    if ret:
                        self.frames[camera_id] = cv2.resize(frame, (self.frame_width, self.frame_height))
                    time.sleep(0.033)  # Limit to ~30 fps
            except Exception as e:
                print(f"Error capturing from camera {camera_id}: {str(e)}")
                time.sleep(1)

    def _start_capture_threads(self):
        """Start capture threads for all cameras"""
        for cam_id in self.cameras.keys():
            thread = threading.Thread(target=self._capture_thread, args=(cam_id,))
            thread.daemon = True
            thread.start()
            self.capture_threads.append(thread)

    def _create_button(self, width: int) -> np.ndarray:
        """Create capture button image"""
        button = np.ones((self.button_height, width, 3), dtype=np.uint8) * 200
        text = "Click here or press SPACE to capture"
        font = cv2.FONT_HERSHEY_SIMPLEX
        textsize = cv2.getTextSize(text, font, 0.7, 2)[0]  # Smaller font
        
        # Center the text
        x = (width - textsize[0]) // 2
        y = (self.button_height + textsize[1]) // 2
        
        cv2.putText(button, text, (x, y), font, 0.7, (0, 0, 0), 2)
        return button

    def capture_images(self):
        """Capture images from all cameras"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for cam_id, frame in self.frames.items():
            if frame is not None:
                filename = f"{self.output_dir}/camera_{cam_id}_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
        
        print(f"Images captured at {timestamp}")

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events"""
        height = param[0]
        if event == cv2.EVENT_LBUTTONDOWN:
            if y > height - self.button_height:
                self.button_pressed = True

    def run(self):
        """Main loop for the application"""
        window_name = "Multi-Camera Capture"
        
        # Calculate grid layout with padding
        padding = 5  # Smaller padding
        
        # Create main canvas
        width = self.frame_width * 2 + padding * 3
        height = self.frame_height * 2 + padding * 3 + self.button_height
        
        # Create window before starting the loop
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window_name, self.mouse_callback, param=[height])
        
        while self.running:
            # Create canvas
            canvas = np.ones((height, width, 3), dtype=np.uint8) * 240
            
            # Draw frames
            positions = [
                (padding, padding),  # Top left
                (self.frame_width + padding * 2, padding),  # Top right
                (padding + self.frame_width//2 + padding//2, self.frame_height + padding * 2)  # Bottom center
            ]
            
            for (cam_id, frame), pos in zip(self.frames.items(), positions):
                if frame is not None:
                    x, y = pos
                    canvas[y:y+self.frame_height, x:x+self.frame_width] = frame
                    
                    # Add camera label
                    label = f"Camera {cam_id}"
                    cv2.putText(canvas, label, (x+5, y+20), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Add button at bottom
            button = self._create_button(width)
            canvas[height-self.button_height:height, 0:width] = button
            
            # Show the canvas
            cv2.imshow(window_name, canvas)
            
            # Handle key events and button press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
            elif key == ord(' ') or self.button_pressed:
                self.capture_images()
                self.button_pressed = False

        # Cleanup
        self._cleanup()

    def _cleanup(self):
        """Cleanup resources"""
        self.running = False
        for thread in self.capture_threads:
            thread.join()
        for cap in self.cameras.values():
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        # Create and run the camera capture system
        capture_system = MultiCameraCapture()
        capture_system.run()
    except KeyboardInterrupt:
        print("\nExiting program...")
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        if 'capture_system' in locals():
            capture_system._cleanup()