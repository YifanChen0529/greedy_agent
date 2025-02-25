"""LIO with policy gradient for policy optimization."""
import numpy as np
import tensorflow as tf

# import lio.alg.networks as networks
from lio.alg import networks
import lio.utils.util as util


class LIO(object):

    def __init__(self, config, l_obs, l_action, nn, agent_name,
                 r_multiplier=2, n_agents=1, agent_id=0):
        self.alg_name = 'lio'
        self.l_obs = l_obs
        self.l_action = l_action
        self.nn = nn
        self.agent_name = agent_name
        self.r_multiplier = r_multiplier
        self.n_agents = n_agents
        self.agent_id = agent_id

        self.list_other_id = list(range(0, self.n_agents))
        del self.list_other_id[self.agent_id]

        # Default is allow the agent to give rewards
        self.can_give = True

        self.entropy_coeff = config.entropy_coeff
        self.gamma = config.gamma
        self.include_cost_in_chain_rule = config.include_cost_in_chain_rule
        self.lr_actor = config.lr_actor
        self.lr_cost = config.lr_cost
        self.lr_reward = config.lr_reward
        if 'optimizer' in config:
            self.optimizer = config.optimizer
        else:
            self.optimizer = 'sgd'
        self.reg = config.reg
        self.reg_coeff = config.reg_coeff
        self.separate_cost_optimizer = config.separate_cost_optimizer

        assert not (self.separate_cost_optimizer and self.include_cost_in_chain_rule)

        self.create_networks()
        self.policy_new = PolicyNew
        
    def create_networks(self):
        self.obs = tf.placeholder(tf.float32, [None, self.l_obs], 'l_obs')
        self.action_others = tf.placeholder(
            tf.float32, [None, self.l_action * (self.n_agents - 1)])
        self.epsilon = tf.placeholder(tf.float32, None, 'epsilon')

        with tf.variable_scope(self.agent_name):
            with tf.variable_scope('policy_main'):
                with tf.variable_scope('policy'):
                    probs = networks.actor_mlp(self.obs, self.l_action, self.nn)
                with tf.variable_scope('eta'):
                    self.reward_function = networks.reward_mlp(self.obs, self.action_others,
                                                               self.nn, n_recipients=self.n_agents)
                self.probs = (1 - self.epsilon) * probs + self.epsilon / self.l_action
                self.log_probs = tf.log(self.probs)
                self.action_samples = tf.multinomial(self.log_probs, 1)

            with tf.variable_scope('policy_prime'):
                with tf.variable_scope('policy'):
                    probs = networks.actor_mlp(self.obs, self.l_action, self.nn)
                self.probs_prime = (1-self.epsilon)*probs + self.epsilon/self.l_action
                self.log_probs_prime = tf.log(self.probs_prime)
                self.action_samples_prime = tf.multinomial(self.log_probs_prime, 1)

        self.policy_params = tf.trainable_variables(
            self.agent_name + '/policy_main/policy')
        self.policy_prime_params = tf.trainable_variables(
            self.agent_name + '/policy_prime/policy')

        self.list_copy_main_to_prime_ops = []
        for idx, var in enumerate(self.policy_prime_params):
            self.list_copy_main_to_prime_ops.append(
                var.assign(self.policy_params[idx]))

        self.list_copy_prime_to_main_ops = []
        for idx, var in enumerate(self.policy_params):
            self.list_copy_prime_to_main_ops.append(
                var.assign(self.policy_prime_params[idx]))

    def receive_list_of_agents(self, list_of_agents):
        self.list_of_agents = list_of_agents

    def run_actor(self, obs, sess, epsilon, prime=False):
        feed = {self.obs: np.array([obs]), self.epsilon: epsilon}
        if prime:
            action = sess.run(self.action_samples_prime, feed_dict=feed)[0][0]
        else:
            action = sess.run(self.action_samples, feed_dict=feed)[0][0]
        return action

    def give_reward(self, obs, action_all, sess):
        action_others_1hot = util.get_action_others_1hot(action_all, self.agent_id,
                                                         self.l_action)
        feed = {self.obs: np.array([obs]),
                self.action_others: np.array([action_others_1hot])}
        reward = sess.run(self.reward_function, feed_dict=feed)
        reward = reward.flatten() * self.r_multiplier

        return reward

    def create_policy_gradient_op(self):
        self.r_ext = tf.placeholder(tf.float32, [None], 'r_ext')

        r2 = self.r_ext
        this_agent_1hot = tf.one_hot(indices=self.agent_id, depth=self.n_agents)
        for other_id in self.list_other_id:
            r2 += self.r_multiplier * tf.reduce_sum(
                tf.multiply(self.list_of_agents[other_id].reward_function,
                            this_agent_1hot), axis=1)

        if self.include_cost_in_chain_rule:
            # for this agent j, subtract the rewards given to all other agents
            # i.e. minus \sum_{i=1}^{N-1} r^i_{eta^j}
            reverse_1hot = 1 - tf.one_hot(indices=self.agent_id, depth=self.n_agents)
            r2 -= self.r_multiplier * tf.reduce_sum(
                tf.multiply(self.reward_function, reverse_1hot), axis=1)

        self.ones = tf.placeholder(tf.float32, [None], 'ones')
        self.gamma_prod = tf.math.cumprod(self.ones * self.gamma)
        returns = tf.reverse(
            tf.math.cumsum(tf.reverse(r2 * self.gamma_prod, axis=[0])), axis=[0])
        returns = returns / self.gamma_prod

        self.action_taken = tf.placeholder(tf.float32, [None, self.l_action],
                                           'action_taken')
        self.log_probs_taken = tf.log(tf.reduce_sum(
            tf.multiply(self.probs, self.action_taken), axis=1) + 1e-15)

        self.entropy = -tf.reduce_sum(tf.multiply(self.probs, self.log_probs))

        self.policy_loss = -tf.reduce_sum(
            tf.multiply(self.log_probs_taken, returns))
        self.loss = self.policy_loss - self.entropy_coeff * self.entropy

        self.policy_grads = tf.gradients(self.loss, self.policy_params)
        grads_and_vars = list(zip(self.policy_grads, self.policy_params))
        self.policy_opt = tf.train.GradientDescentOptimizer(self.lr_actor)
        self.policy_op = self.policy_opt.apply_gradients(grads_and_vars)

    def create_update_op(self):
        self.r_from_others = tf.placeholder(tf.float32, [None], 'r_from_others')
        r2_val = self.r_ext + self.r_from_others
        if self.include_cost_in_chain_rule:
            self.r_given = tf.placeholder(tf.float32, [None], 'r_given')
            r2_val -= self.r_given
        returns_val = tf.reverse(tf.math.cumsum(
            tf.reverse(r2_val * self.gamma_prod, axis=[0])), axis=[0])
        returns_val = returns_val / self.gamma_prod

        log_probs_taken = tf.log(tf.reduce_sum(
            tf.multiply(self.probs_prime, self.action_taken), axis=1) + 1e-15)
        entropy = -tf.reduce_sum(
            tf.multiply(self.probs_prime, self.log_probs_prime))
        policy_loss = -tf.reduce_sum(
            tf.multiply(log_probs_taken, returns_val))
        loss = policy_loss - self.entropy_coeff * entropy

        policy_opt_prime = tf.train.GradientDescentOptimizer(self.lr_actor)
        self.policy_op_prime = policy_opt_prime.minimize(loss)

    def create_reward_train_op(self):
        list_reward_loss = []
        self.list_policy_new = [0 for x in range(self.n_agents)]
        self.returns = tf.placeholder(tf.float32, [None], 'returns')

        for agent in self.list_of_agents:
            if agent.agent_id == self.agent_id and not self.include_cost_in_chain_rule:
                # In this case, cost for giving is not accounted in chain rule,
                # so the agent can skip over itself
                continue
            other_policy_params_new = {}
            for grad, var in zip(agent.policy_grads, agent.policy_params):
                other_policy_params_new[var.name] = var - agent.lr_actor * grad
            other_policy_new = agent.policy_new(
                other_policy_params_new, agent.l_obs, agent.l_action,
                agent.agent_name)
            self.list_policy_new[agent.agent_id] = other_policy_new

            log_probs_taken = tf.log(
                tf.reduce_sum(tf.multiply(other_policy_new.probs,
                                          other_policy_new.action_taken), axis=1))
            loss_term = -tf.reduce_sum(tf.multiply(log_probs_taken, self.returns))
            list_reward_loss.append(loss_term)

        if self.include_cost_in_chain_rule:
            self.reward_loss = tf.reduce_sum(list_reward_loss)
        else:
            reverse_1hot = 1 - tf.one_hot(indices=self.agent_id, depth=self.n_agents)
            if self.separate_cost_optimizer or self.reg == 'l1':
                given_each_step = tf.reduce_sum(tf.abs(
                    tf.multiply(self.reward_function, reverse_1hot)), axis=1)
                total_given = tf.reduce_sum(tf.multiply(
                    given_each_step, self.gamma_prod/self.gamma))
            elif self.reg == 'l2':
                total_given = tf.reduce_sum(tf.square(
                    tf.multiply(self.reward_function, reverse_1hot)))
            if self.separate_cost_optimizer:
                self.reward_loss = tf.reduce_sum(list_reward_loss)
            else:
                self.reward_loss = (tf.reduce_sum(list_reward_loss) +
                                    self.reg_coeff * total_given)
            
        if self.optimizer == 'sgd':
            reward_opt = tf.train.GradientDescentOptimizer(self.lr_reward)
            if self.separate_cost_optimizer:
                cost_opt = tf.train.GradientDescentOptimizer(self.lr_cost)
        elif self.optimizer == 'adam':
            reward_opt = tf.train.AdamOptimizer(self.lr_reward)
            if self.separate_cost_optimizer:
                cost_opt = tf.train.AdamOptimizer(self.lr_cost)
        self.reward_op = reward_opt.minimize(self.reward_loss)
        if self.separate_cost_optimizer:
            self.cost_op = cost_opt.minimize(total_given)

    def update(self, sess, buf, epsilon):
        sess.run(self.list_copy_main_to_prime_ops)

        n_steps = len(buf.obs)
        actions_1hot = util.process_actions(buf.action, self.l_action)
        ones = np.ones(n_steps)
        feed = {self.obs: buf.obs,
                self.action_taken: actions_1hot,
                self.r_ext: buf.reward,
                self.ones: ones,
                self.epsilon: epsilon}
        sum_r_from_other = []
        # print(buf.r_from_others)
        for reward in buf.r_from_others:
            temp = np.sum(reward, axis=0, keepdims=False)
            sum_r_from_other.append(temp[self.agent_id])
        # print(sum_r_from_other)
        feed[self.r_from_others] = sum_r_from_other
        # feed[self.r_from_others] = buf.r_from_others
        if self.include_cost_in_chain_rule:
            feed[self.r_given] = buf.r_given

        _ = sess.run(self.policy_op_prime, feed_dict=feed)

    def train_reward(self, sess, list_buf, list_buf_new, epsilon,
                     summarize=False, writer=None):
        buf_self = list_buf[self.agent_id]
        buf_self_new = list_buf_new[self.agent_id]

        n_steps = len(buf_self.obs)
        ones = np.ones(n_steps)

        feed = {}

        for agent in self.list_of_agents:
            other_id = agent.agent_id
            if other_id == self.agent_id:
                continue
            buf_other = list_buf[other_id]
            actions_other_1hot = util.process_actions(buf_other.action, self.l_action)
            feed[agent.obs] = buf_other.obs
            feed[agent.action_taken] = actions_other_1hot
            feed[agent.r_ext] = buf_other.reward
            feed[agent.ones] = ones
            feed[agent.epsilon] = epsilon

            # This is needed for the case N > 2. From an agent i's perspective,
            # another agent j will receive reward from a third agent k, 
            # so to compute j's policy update we need to input agent k's observation
            # of all other agents' actions (from agent k's perspective).
            # So in general we just feed action_others from all agents' perspectives.
            feed[agent.action_others] = util.get_action_others_1hot_batch(
                buf_other.action_all, other_id, agent.l_action)

            buf_other_new = list_buf_new[other_id]
            actions_other_1hot_new = util.process_actions(buf_other_new.action,
                                                          self.l_action)
            other_policy_new = self.list_policy_new[other_id]
            feed[other_policy_new.obs] = buf_other_new.obs
            feed[other_policy_new.action_taken] = actions_other_1hot_new

        if self.include_cost_in_chain_rule:
            # Needed to compute the chain rule,
            # These are for the update from \theta to \hat{\theta}
            action_self_1hot = util.process_actions(buf_self.action, self.l_action)
            feed[self.action_taken] = action_self_1hot
            feed[self.r_ext] = buf_self.reward
            feed[self.epsilon] = epsilon
            # These are needed for the factor
            # \nabla_{\hat{\theta}^j} J^i(\hat{\tau}, \hat{\theta}) when i == j
            action_self_1hot_new = util.process_actions(buf_self_new.action,
                                                        self.l_action)
            self_policy_new = self.list_policy_new[self.agent_id]
            feed[self_policy_new.obs] = buf_self_new.obs
            feed[self_policy_new.action_taken] = action_self_1hot_new

        n_steps = len(buf_self_new.obs)
        if self.include_cost_in_chain_rule:
            total_reward = [buf_self_new.reward[idx] + buf_self_new.r_from_others[idx]
                            - buf_self_new.r_given[idx] for idx in range(n_steps)]
        else:
            total_reward = buf_self_new.reward
        returns_new = util.process_rewards(total_reward, self.gamma)
        feed[self.obs] = buf_self.obs
        feed[self.action_others] = util.get_action_others_1hot_batch(
            buf_self.action_all, self.agent_id, self.l_action)
        feed[self.ones] = ones

        feed[self.returns] = returns_new

        if self.separate_cost_optimizer:
            _ = sess.run([self.reward_op, self.cost_op], feed_dict=feed)
        else:
            _ = sess.run(self.reward_op, feed_dict=feed)

    def update_main(self, sess):
        sess.run(self.list_copy_prime_to_main_ops)

    def set_can_give(self, can_give):
        self.can_give = can_give


class PolicyNew(object):
    def __init__(self, params, l_obs, l_action, agent_name):
        self.obs = tf.placeholder(tf.float32, [None, l_obs], 'obs_new')
        self.action_taken = tf.placeholder(tf.float32, [None, l_action],
                                           'action_taken')
        prefix = agent_name + '/policy_main/policy/'
        with tf.variable_scope('policy_new'):
            h1 = tf.nn.relu(
                tf.nn.xw_plus_b(self.obs, params[prefix + 'actor_h1/kernel:0'],
                                params[prefix + 'actor_h1/bias:0']))
            h2 = tf.nn.relu(
                tf.nn.xw_plus_b(h1, params[prefix + 'actor_h2/kernel:0'],
                                params[prefix + 'actor_h2/bias:0']))
            out = tf.nn.xw_plus_b(h2, params[prefix + 'actor_out/kernel:0'],
                                params[prefix + 'actor_out/bias:0'])
        self.probs = tf.nn.softmax(out)

        