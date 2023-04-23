#!/home/shb/miniconda3/envs/lio/bin python3

import numpy as np
import tensorflow as tf
import random
import os

import argparse
import time 

from lio.alg import config_room_lio
from lio.env import room_symmetric
from lio_agent import LIO


import rospy
from std_msgs.msg import String




sync_flag = False

def my_req(data):
    if sync_flag:
        agent_sync_pub[i].publish('req')

def agent1_clback(data):
    agent1_data = data.data
    agent_sync_flag[1] = True


def agent_sync(agents_num):
    global sync_flag
    sync_flag = True
    other_agents = []
    for i in range(1,agents_num+1):
        agent_sync_flag[i] = False
        agent_sync_pub[i].publish('req')

        while not agent_sync_flag[i]:
            time.sleep(0.01)
        
        other_agents.append(data)
    
    sync_flag = False
    return other_agents

        

def get_agent_list_buffer():
    pass
def get_list_buffers_new():
    pass
def init_agent():
    seed = config.main.seed
    np.random.seed(seed)
    random.seed(seed)
    tf.set_random_seed(seed)

    dir_name = config.main.dir_name
    exp_name = config.main.exp_name
    log_path = os.path.join('..', 'results', exp_name, dir_name)
    model_name = config.main.model_name
    save_period = config.main.save_period


    n_episodes = int(config.alg.n_episodes)
    n_eval = config.alg.n_eval
    period = config.alg.period

    epsilon = config.lio.epsilon_start
    epsilon_step = (
        epsilon - config.lio.epsilon_end) / config.lio.epsilon_div


    env = room_symmetric.Env(config.env) # TODO: remove

    agent = LIO(config.lio, env.l_obs, env.l_action,
                            config.nn, 'agent_0',
                            config.env.r_multiplier, env.n_agents)

    other_agents = agent_sync()
    agent.create_policy_gradient_op(other_agents)#
    agent.create_update_op()

    other_agents = agent_sync()
    agent.create_reward_train_op(other_agents)#


    config_proto = tf.ConfigProto()
    if config.main.use_gpu:
        config_proto.device_count['GPU'] = 1
        config_proto.gpu_options.allow_growth = True
    else:
        config_proto.device_count['GPU'] = 0
    sess = tf.Session(config=config_proto)
    sess.run(tf.global_variables_initializer())

    list_agent_meas = []
    if config.env.name == 'er':
        list_suffix = ['reward_total', 'n_lever', 'n_door',
                       'received', 'given', 'r-lever', 'r-start', 'r-door']

    for agent_id in range(1, env.n_agents + 1):
        for suffix in list_suffix:
            list_agent_meas.append('A%d_%s' % (agent_id, suffix))

    saver = tf.train.Saver(max_to_keep=config.main.max_to_keep)

    agent_sync()

    list_buffers = get_agent_list_buffer()
    agent.update(sess, list_buffers, epsilon)

    other_agents = agent_sync()
    list_buffers_new = get_list_buffers_new()
    if agent.can_give:
        agent.train_reward(sess, list_buffers,
                            list_buffers_new, epsilon, other_agents)#

    agent_sync()

    agent.update_main(sess)

def talker():
    rate = rospy.Rate(100) # 100hz
    while not rospy.is_shutdown():
        hello_str = "hello world %s" % rospy.get_time()
        rospy.loginfo(hello_str)
        pub.publish(hello_str)
        rate.sleep()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('id', type=int)
    args = parser.parse_args()
    try:

        rospy.init_node("agent%d"%(args.id), anonymous=False)
        pub = rospy.Publisher('chatter', String, queue_size=10)
        rospy.Subscriber("agent%d/req"%(args.id), String, my_req)
        rospy.Subscriber("agent1/data", String, agent1_clback)
        config = config_room_lio.get_config()
        init_agent()
        talker()
    except rospy.ROSInterruptException:
        pass