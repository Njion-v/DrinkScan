import numpy as np
import cv2

class CameraCalibrationExplainer:
    def __init__(self):
        """
        Khởi tạo các tham số camera calibration
        """
        # Tham số nội (Intrinsic Parameters)
        self.camera_matrix = None  # Ma trận camera (focal length, optical centers)
        self.dist_coeffs = None    # Hệ số biến dạng
        
        # Tham số ngoại (Extrinsic Parameters)
        self.rotation_vectors = None    # Vector quay
        self.translation_vectors = None # Vector dịch chuyển
        
    def calibrate_camera(self, images, pattern_size, square_size):
        """
        Thực hiện calibration camera sử dụng mẫu điểm tròn
        
        Parameters:
        - images: Danh sách ảnh chụp mẫu điểm
        - pattern_size: (rows, cols) của mẫu điểm
        - square_size: Khoảng cách thực tế giữa các điểm (mm)
        """
        # Chuẩn bị điểm trong không gian 3D thực tế
        object_points = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        object_points[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        object_points *= square_size
        
        # Lists để lưu điểm 3D và 2D từ tất cả ảnh
        obj_points = []  # điểm 3D
        img_points = []  # điểm 2D
        
        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Tìm các điểm tròn trong ảnh
            ret, corners = cv2.findCirclesGrid(
                gray, 
                pattern_size, 
                cv2.CALIB_CB_SYMMETRIC_GRID
            )
            
            if ret:
                obj_points.append(object_points)
                img_points.append(corners)
        
        # Thực hiện calibration
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj_points,
            img_points,
            gray.shape[::-1],
            None,
            None
        )
        
        self.camera_matrix = mtx
        self.dist_coeffs = dist
        self.rotation_vectors = rvecs
        self.translation_vectors = tvecs
        
        return {
            'camera_matrix': mtx,        # Ma trận camera
            'dist_coeffs': dist,         # Hệ số biến dạng
            'rotation_vectors': rvecs,    # Vector quay
            'translation_vectors': tvecs  # Vector dịch chuyển
        }

    def explain_parameters(self):
        """
        Giải thích ý nghĩa các tham số calibration
        """
        if self.camera_matrix is not None:
            fx = self.camera_matrix[0,0]  # Focal length X
            fy = self.camera_matrix[1,1]  # Focal length Y
            cx = self.camera_matrix[0,2]  # Principal point X
            cy = self.camera_matrix[1,2]  # Principal point Y
            
            return {
                'focal_length': {
                    'fx': fx,
                    'fy': fy,
                    'explanation': 'Độ dài tiêu cự theo trục X và Y'
                },
                'principal_point': {
                    'cx': cx,
                    'cy': cy,
                    'explanation': 'Tâm quang học của camera'
                },
                'distortion_coefficients': {
                    'values': self.dist_coeffs,
                    'explanation': 'Hệ số biến dạng hình ảnh (radial và tiếp tuyến)'
                }
            }
        return None