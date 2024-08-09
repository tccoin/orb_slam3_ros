#!/usr/bin/env python

import rospy
import tf2_ros
import numpy as np
from geometry_msgs.msg import PoseStamped
from tf2_msgs.msg import TFMessage

def quat_to_rot_matrix(q):
    """
    Converts a quaternion to a rotation matrix.
    
    Parameters:
    q (numpy.array): A quaternion [qx, qy, qz, qw]

    Returns:
    numpy.array: A 3x3 rotation matrix
    """
    qx, qy, qz, qw = q
    # Calculate the elements of the rotation matrix
    R = np.array([
        [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)]
    ])
    return R

def rot_matrix_to_quat(R):
    """
    Converts a rotation matrix to a quaternion.
    
    Parameters:
    R (numpy.array): A 3x3 rotation matrix

    Returns:
    numpy.array: A quaternion [qx, qy, qz, qw]
    """
    # Ensure the matrix is 3x3
    assert R.shape == (3, 3)
    
    m00, m01, m02 = R[0, :]
    m10, m11, m12 = R[1, :]
    m20, m21, m22 = R[2, :]
    
    trace = m00 + m11 + m22

    if trace > 0:
        S = 2.0 * np.sqrt(trace + 1.0)
        qw = 0.25 * S
        qx = (m21 - m12) / S
        qy = (m02 - m20) / S
        qz = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
        qw = (m21 - m12) / S
        qx = 0.25 * S
        qy = (m01 + m10) / S
        qz = (m02 + m20) / S
    elif m11 > m22:
        S = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
        qw = (m02 - m20) / S
        qx = (m01 + m10) / S
        qy = 0.25 * S
        qz = (m12 + m21) / S
    else:
        S = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
        qw = (m10 - m01) / S
        qx = (m02 + m20) / S
        qy = (m12 + m21) / S
        qz = 0.25 * S
    
    return np.array([qx, qy, qz, qw])

class TumTrajectoryPublisher:
    def __init__(self):
        self.trajectory = self.load_trajectory('/home/link/.ros/traj.tum')
        self.br = tf2_ros.TransformBroadcaster()
        self.pose_sub = rospy.Subscriber('/orb_slam3/camera_pose', PoseStamped, self.pose_callback)

    def load_trajectory(self, file_path):
        trajectory = {}
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 8:
                    timestamp = float(parts[0])
                    pose = {
                        'tx': float(parts[1]),
                        'ty': float(parts[2]),
                        'tz': float(parts[3]),
                        'qx': float(parts[4]),
                        'qy': float(parts[5]),
                        'qz': float(parts[6]),
                        'qw': float(parts[7])
                    }
                    trajectory[timestamp] = pose
        return trajectory

    def pose_callback(self, msg):
        timestamp = msg.header.stamp.to_sec()
        if timestamp in self.trajectory:
            pose = self.trajectory[timestamp]
            self.publish_tf(pose, msg.header.stamp)

    def publish_tf(self, pose, stamp):
        t = TFMessage()
        transform = tf2_ros.TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = "head_camera_rgb_optical_frame"
        transform.child_frame_id = "orb_slam_loop_closure"

        T = np.eye(4)
        T[:3, 3] = np.array([pose['tx'], pose['ty'], pose['tz']])
        T[:3, :3] = quat_to_rot_matrix([pose['qx'], pose['qy'], pose['qz'], pose['qw']])
        T = np.linalg.inv(T)
        quat = rot_matrix_to_quat(T[:3, :3])

        transform.transform.translation.x = T[0, 3]
        transform.transform.translation.y = T[1, 3]
        transform.transform.translation.z = T[2, 3]
        transform.transform.rotation.x = quat[0]
        transform.transform.rotation.y = quat[1]
        transform.transform.rotation.z = quat[2]
        transform.transform.rotation.w = quat[3]
        t.transforms.append(transform)
        self.br.sendTransform(transform)

if __name__ == '__main__':
    rospy.init_node('tum_trajectory_publisher')
    node = TumTrajectoryPublisher()
    rospy.spin()