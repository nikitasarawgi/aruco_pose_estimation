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
            self.detect_aruco,  # The callback function needs to be implemented
            10
        )
        self.bridge = CvBridge()
        self.arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.arucoParams = cv2.aruco.DetectorParameters()
        self.markerLength = 0.0175

        self.cameraIntrinsicsFile = ""
        self.cameraIntrinsics = self.read_camera_intrinsics(self.cameraIntrinsicsFile)

        self.distCoeffFile = ""
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

        frame = self.bridge.imgmsg_to_cv2(image, desired_encoding='bgr8')

        arucoDetector = cv2.aruco.ArucoDetector(self.arucoDict, self.arucoParams)
        (corners, ids, rejected) = arucoDetector.detectMarkers(image)
        if ids is not None:
            objPoints = self.get_marker_3d_points(self.markerLength)

            rvecs = []
            tvecs = []

            for corner in corners:
                retval, rvec, tvec = cv2.solvePnP(objectPoints=objPoints,
                                                  imagePoints=corner,
                                                   cameraMatrix=self.cameraIntrinsics,
                                                    distCoeffs=self.distCoeff)
                if retval:
                    rvecs.append(rvec)
                    tvecs.append(tvec)
                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                    cv2.drawFrameAxes(frame, self.cameraIntrinsics, self.distCoeff, rvec, tvec, self.markerLength)
        print(rvecs)
        print(tvecs)
        cv2.imshow('Aruco Markers', frame)
        cv2.waitKey(1)



def main(args=None):
    rclpy.init(args=args)

    arucoPoseEstimator = ArucoPoseEstimator()

    rclpy.spin(arucoPoseEstimator) 
    arucoPoseEstimator.destroy_node()
    rclpy.shutdown()
            




