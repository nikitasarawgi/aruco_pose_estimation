import rclpy
from rclpy.node import Node
from sensor_msgs.msg._image import Image
from sensor_msgs.msg._camera_info import CameraInfo
import cv2
import numpy as np
from cv_bridge import CvBridge

class ArucoPoseEstimator(Node):
    def __init__(self):
        super().__init__('aruco_pose_estimator')
        self.imageTopic = "/camera/camera/color/image_raw"
        self.imageStream = self.create_subscription(
            Image, 
            self.imageTopic, 
            self.detect_aruco,  
            10
        )
        self.bridge = CvBridge()
        self.arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.arucoParams = cv2.aruco.DetectorParameters()
        self.markerLength = 0.0175

        self.cameraIntrinsicsFile = "/home/aero/nisara/aruco_pose_estimation/src/aruco_pose_estimation/camera_intrinsics.txt"
        self.cameraIntrinsics = self.read_camera_intrinsics(self.cameraIntrinsicsFile)

        self.distCoeffFile = "/home/aero/nisara/aruco_pose_estimation/src/aruco_pose_estimation/distortion_coeffs.txt"
        self.distCoeff = self.read_dist_coeff(self.distCoeffFile)

    def get_marker_3d_points(self, markerLength):
        halfSize = markerLength / 2.0
        return np.array([
            [-halfSize, halfSize, 0],
            [halfSize, halfSize, 0],
            [halfSize, -halfSize, 0],
            [-halfSize, -halfSize, 0]
        ], dtype=np.float32)

    def read_camera_intrinsics(self, filePath):
        with open(filePath, 'r') as file:
            lines = file.readlines()
            matrix = np.array([list(map(float, line.split())) for line in lines])
        return matrix
    
    def read_dist_coeff(self, filePath):
        with open(filePath, 'r') as file:
            coeffs = list(map(float, file.read().split()))
        return np.array(coeffs)
    
    
    def detect_aruco(self, image):

        print(type(image))

        rvecs = {}
        tvecs = {}
        

        frame = self.bridge.imgmsg_to_cv2(image, desired_encoding='bgr8')

        arucoDetector = cv2.aruco.ArucoDetector(self.arucoDict, self.arucoParams)
        (corners, ids, rejected) = arucoDetector.detectMarkers(frame)
        if ids is not None:
            objPoints = self.get_marker_3d_points(self.markerLength)

            rvecs = {}
            tvecs = {}

            for id, corner in zip(ids, corners):
                retval, rvec, tvec = cv2.solvePnP(objectPoints=objPoints,
                                                  imagePoints=corner,
                                                   cameraMatrix=self.cameraIntrinsics,
                                                    distCoeffs=self.distCoeff)
                if retval:
                    rvecs[id[0]] = rvec
                    tvecs[id[0]] = tvec
                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                    cv2.drawFrameAxes(frame, self.cameraIntrinsics, self.distCoeff, rvec, tvec, self.markerLength)
            print("RVECS: ", rvecs)
            print("TVECS: ", tvecs)
            
            center_position = np.mean([tvec for tvec in tvecs.values()], axis = 0)
            print("Center position: ", center_position)

            # # Plot this center only
            # image_points, _ = cv2.projectPoints(center_position, np.zeros((3,1)), np.zeros((3,1)), self.cameraIntrinsics, self.distCoeff)
            # print("Image Points: ", image_points)
            # center_2d = tuple(image_points[0].ravel().astype(int))
            # cv2.circle(frame, center_2d, radius = 5, color = (0, 0, 255), thickness=-1)

            required_ids = [5, 15, 25, 35]
            if all(key in tvecs for key in required_ids):

                vector_x = (tvecs[25] - tvecs[35]).ravel()
                vector_y = (tvecs[5] - tvecs[15]).ravel()
                vector_x /= np.linalg.norm(vector_x)
                vector_y /= np.linalg.norm(vector_y)
                vector_z = np.cross(vector_x, vector_y)
                vector_z /= np.linalg.norm(vector_z)

                center_orientation_matrix = np.column_stack((vector_x, vector_y, vector_z))
                center_orientation, _ = cv2.Rodrigues(center_orientation_matrix)
                print("Center orientation: ", center_orientation)

                cv2.drawFrameAxes(frame, self.cameraIntrinsics, self.distCoeff, center_orientation, center_position, self.markerLength)

            

        cv2.imshow('Aruco Markers', frame)
        cv2.waitKey(1)



def main(args=None):
    rclpy.init(args=args)

    arucoPoseEstimator = ArucoPoseEstimator()

    rclpy.spin(arucoPoseEstimator) 
    arucoPoseEstimator.destroy_node()
    rclpy.shutdown()
            




