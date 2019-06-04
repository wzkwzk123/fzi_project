#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy, sys
import moveit_commander
from moveit_msgs.msg import RobotTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

from geometry_msgs.msg import PoseStamped, Pose
from tf.transformations import euler_from_quaternion, quaternion_from_euler

class MoveItIkDemo:
    def __init__(self):
        # 初始化move_group的API
        moveit_commander.roscpp_initialize(sys.argv)
        
        # 初始化ROS节点
        rospy.init_node('moveit_ik_demo')
                
        # 初始化需要使用move group控制的机械臂中的arm group
        hand = moveit_commander.MoveGroupCommander('hand')
        arm = moveit_commander.MoveGroupCommander('arm')
                
        # 获取终端link的名称
        
        arm.set_named_target('start')
        arm.go()
        rospy.sleep(2)

        hand.set_named_target('close')
        hand.go()
        rospy.sleep(2)

        hand.set_named_target('open')
        hand.go()
        rospy.sleep(2)


               
        
         
        
           
        

        # 关闭并退出moveit
        moveit_commander.roscpp_shutdown()
        moveit_commander.os._exit(0)

if __name__ == "__main__":
    MoveItIkDemo()

    
    
