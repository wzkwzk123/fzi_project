#!/usr/bin/env python

import gym
import numpy
import random
import time
import qlearn
from gym import wrappers
# ROS packages required
import rospy
import rospkg
# import our training environment
from openai_ros.task_envs.shadow_tc import learn_to_pick_ball

import json

nepisodes = 500
nsteps = 1000
def my_chooseaAction(state):
    actions=range(env.action_space.n)
    q = [getQ(state, a) for a in actions]
    maxQ = max(q)

    count = q.count(maxQ)
    if count > 1:
        best = [i for i in range(len(actions)) if q[i] == maxQ]
        i = random.choice(best)
    else:
        i = q.index(maxQ)

    action = actions[i]
    return action






def getQ(state, action):
    return q_value.get((state, action), 0.0)


if __name__ == '__main__':

    rospy.init_node('shadow_tc_learn_to_pick_ball_qlearn', anonymous=True, log_level=rospy.WARN)

    # Create the Gym environment
    env = gym.make('ShadowTcGetBall-v0')
    # Set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('my_shadow_tc_openai_example')
    outdir = pkg_path + '/training_results'
    env = wrappers.Monitor(env, outdir, force=True)
    rospy.loginfo("Monitor Wrapper started")

    last_time_steps = numpy.ndarray(0)

    start_time = time.time()
    highest_reward = 0

    f = open('/home/wzk/catkin_ws/src/new_start/my_shadow_tc_openai_example/scripts/result.txt','r')
    q_value = f.read()
    f.close()
    q_value = eval(q_value)






    # Starts the main training loop: the one about the episodes to do
    for x in range(nepisodes):  # 500
        rospy.logdebug("############### START EPISODE=>" + str(x))

        cumulated_reward = 0
        done = False

        # Initialize the environment and get first state of the robot
        observation = env.reset()
        state = ''.join(map(str, observation)) # string process, the problem is state space is too large

        # Show on screen the actual situation of the robot
        # env.render()
        # for each episode, we test the robot for nsteps
        for i in range(nsteps):
            # Pick an action based on the current state
            action = my_chooseaAction(state)
            # Execute the action in the environment and get feedback
            observation, reward, done, info = env.step(action)


            nextState = ''.join(map(str, observation))


            if not (done):
                state = nextState
            else:
                rospy.logwarn("DONE")
                last_time_steps = numpy.append(last_time_steps, [int(i + 1)])
                break
            #raw_input("Next Step...PRESS KEY")
            # rospy.sleep(

        
        # jsObj = json.dumps(q_value)
        # fileObject = open('q_value.json', 'w')
        # fileObject.write(jsObj)
        # fileObject.close() 


    rospy.loginfo(("\n|" + str(nepisodes) + "|" + str(qlearn.alpha) + "|" + str(qlearn.gamma) + "|" + str(
        initial_epsilon) + "*" + str(epsilon_discount) + "|" + str(highest_reward) + "| PICTURE |"))

    l = last_time_steps.tolist()
    l.sort()

    env.close()