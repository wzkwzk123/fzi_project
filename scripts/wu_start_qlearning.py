#!/usr/bin/env python

import gym
import numpy
import qlearn
from gym import wrappers

import rospy
import rospkg

from openai_ros.task_envs.shadow_tc import learn_to_pick_ball

if __name__ = '__main__' :
	rospy.init_node('wu_shadow_tc_learn_to_pick_ball_qlearn', anonymous = True, log_level=rospy.WARN)