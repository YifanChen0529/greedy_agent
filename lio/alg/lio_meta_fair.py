"""Meta-Fair Learning to Incentivize Others (Meta-Fair-LIO) implementation.
Extends LIO with meta-learning and energy fairness components."""

import numpy as np
import tensorflow as tf
from lio.alg import networks
import lio.utils.util as util

class MetaFairLIO(object):
    """Meta-Fair LIO built on policy gradient."""

    def __init__(self, config, l_obs, l_action, nn, agent_name, 
                 r_multiplier=2, n_agents=1, agent_id=0, energy_param=1.0):
        """Initialize Meta-Fair LIO agent.
        
        Args:
            config: Configuration object
            l_obs: Length of observation
            l_action: Length of action space
            nn: Neural network parameters
            agent_name: Name of agent
            r_multiplier: Reward multiplier
            n_agents: Number of agents
            agent_id: ID of this agent
            energy_param: Energy cost parameter
        """
        self.alg_name = 'meta-fair-lio'
        self.l_obs = l_obs
        self.l_action = l_action
        self.nn = nn
        self.agent_name = agent_name
        self.r_multiplier = r_multiplier
        self.n_agents = n_agents
        self.agent_id = agent_id
        self.energy_param = energy_param

        # Meta-learning parameters
        self.meta_lr = config.meta_lr if hasattr(config, 'meta_lr') else 0.001
        self.meta_batch_size = config.meta_batch_size if hasattr(config, 'meta_batch_size') else 32
        
        # Energy fairness parameters
        self.energy_threshold = config.energy_threshold if hasattr(config, 'energy_threshold') else 0.1
        self.reward_threshold = config.reward_threshold if hasattr(config, 'reward_threshold') else 0.1
        self.energy_penalty = config.energy_penalty if hasattr(config, 'energy_penalty') else 0.01

        # Core LIO parameters from parent
        self.entropy_coeff = config.entropy_coeff
        self.gamma = config.gamma
        self.lr_actor = config.lr_actor
        self.lr_reward = config.lr_reward
        self.reg = config.reg
        self.reg_coeff = config.reg_coeff
        
        self.list_other_id = list(range(0, self.n_agents))
        del self.list_other_id[self.agent_id]
        self.can_give = True # Default allowing reward giving
        
        # Create network components
        self.create_networks()
        self.create_meta_networks()
        self.policy_new = PolicyNew

    def create_networks(self):
        """Create primary policy and reward networks."""
        self.obs = tf.placeholder(tf.float32, [None, self.l_obs], 'l_obs')
        self.action_others = tf.placeholder(
            tf.float32, [None, self.l_action * (self.n_agents - 1)])
        self.epsilon = tf.placeholder(tf.float32, None, 'epsilon')

        with tf.variable_scope(self.agent_name):
            with tf.variable_scope('policy_main'):
                with tf.variable_scope('policy'):
                    probs = networks.actor_mlp(self.obs, self.l_action, self.nn)
                with tf.variable_scope('eta'):
                    self.reward_function = networks.reward_mlp(
                        self.obs, self.action_others, self.nn, 
                        n_recipients=self.n_agents)
                    
                self.probs = (1 - self.epsilon) * probs + self.epsilon / self.l_action
                self.log_probs = tf.log(self.probs)
                self.action_samples = tf.multinomial(self.log_probs, 1)

            # Create prime networks
            with tf.variable_scope('policy_prime'):
                with tf.variable_scope('policy'):
                    probs = networks.actor_mlp(self.obs, self.l_action, self.nn)
                self.probs_prime = (1-self.epsilon)*probs + self.epsilon/self.l_action
                self.log_probs_prime = tf.log(self.probs_prime)
                self.action_samples_prime = tf.multinomial(self.log_probs_prime, 1)

        # Get trainable variables
        self.policy_params = tf.trainable_variables(
            self.agent_name + '/policy_main/policy')
        self.policy_prime_params = tf.trainable_variables(
            self.agent_name + '/policy_prime/policy')

        # Create copy operations
        self.list_copy_main_to_prime_ops = []
        for idx, var in enumerate(self.policy_prime_params):
            self.list_copy_main_to_prime_ops.append(
                var.assign(self.policy_params[idx]))

        self.list_copy_prime_to_main_ops = []
        for idx, var in enumerate(self.policy_params):
            self.list_copy_prime_to_main_ops.append(
                var.assign(self.policy_prime_params[idx]))

    def create_meta_networks(self):
        """Create meta-learning networks."""
        with tf.variable_scope(self.agent_name):
            with tf.variable_scope('meta'):
                # Meta network for policy adaptation
                self.meta_policy = networks.actor_mlp(
                    self.obs, self.l_action, self.nn)
                
                # Meta network for reward function adaptation
                self.meta_reward = networks.reward_mlp(
                    self.obs, self.action_others, self.nn,
                    n_recipients=self.n_agents)

        # Get meta parameters
        self.meta_params = tf.trainable_variables(
            self.agent_name + '/meta')

    def calculate_energy_fairness_loss(self, energies, rewards):
        """Calculate the energy fairness loss component.
        
        Args:
            energies: List of energy consumptions for each agent
            rewards: List of rewards received by each agent
            
        Returns:
            Energy fairness loss value
        """
        # Calculate average energy and reward/energy ratio
        avg_energy = tf.reduce_mean(energies)
        reward_per_energy = rewards / (energies + 1e-8)
        avg_reward_per_energy = tf.reduce_mean(reward_per_energy)

        # Calculate squared differences from averages
        energy_diffs = tf.square(energies - avg_energy) 
        reward_diffs = tf.square(reward_per_energy - avg_reward_per_energy)

        # Total fairness loss
        fairness_loss = tf.reduce_mean(energy_diffs + reward_diffs)
        
        return fairness_loss

    def create_meta_objective(self):
        """Create the meta-learning objective."""
        # Get inputs for meta-learning
        self.meta_rewards = tf.placeholder(tf.float32, [None], 'meta_rewards')
        self.meta_energies = tf.placeholder(tf.float32, [None], 'meta_energies')
        
        # Calculate adaptation objective
        adaptation_loss = -tf.reduce_mean(self.meta_rewards)
        
        # Calculate fairness loss
        fairness_loss = self.calculate_energy_fairness_loss(
            self.meta_energies, self.meta_rewards)
        
        # Combined meta objective
        self.meta_loss = adaptation_loss + self.energy_penalty * fairness_loss
        
        # Create meta optimizer
        meta_opt = tf.train.AdamOptimizer(self.meta_lr)
        self.meta_train_op = meta_opt.minimize(
            self.meta_loss, var_list=self.meta_params)

    def run_actor(self, obs, sess, epsilon, prime=False):
        """Run actor to select actions."""
        feed = {self.obs: np.array([obs]), self.epsilon: epsilon}
        if prime:
            action = sess.run(self.action_samples_prime, feed_dict=feed)[0][0]
        else:
            action = sess.run(self.action_samples, feed_dict=feed)[0][0]
        
        # Calculate energy cost
        energy_cost = self.calculate_energy_cost(obs, action)
        return action

    def calculate_energy_cost(self, state, action):
        """Calculate energy cost of an action.
        
        Args:
            state: Current state
            action: Action to be taken
            
        Returns:
            Energy cost value
        """
        # Energy cost proportional to action magnitude and state complexity
        return self.energy_param * np.abs(action) * (1 + state.mean())

    def update(self, sess, buf, epsilon):
        """Update policy parameters."""
        # Standard LIO policy update
        sess.run(self.list_copy_main_to_prime_ops)

        n_steps = len(buf.obs)
        actions_1hot = util.process_actions(buf.action, self.l_action)
        ones = np.ones(n_steps)
        
        feed = {
            self.obs: buf.obs,
            self.action_taken: actions_1hot,
            self.r_ext: buf.reward,
            self.ones: ones,
            self.epsilon: epsilon
        }

        # Add energy costs to feed
        feed[self.meta_energies] = buf.energy_cost
        
        # Compute rewards from others
        sum_r_from_other = []
        for reward in buf.r_from_others:
            temp = np.sum(reward, axis=0, keepdims=False)
            sum_r_from_other.append(temp[self.agent_id])
        
        if len(sum_r_from_other) < len(buf.reward):
            padding = [0] * (len(buf.reward) - len(sum_r_from_other))
            sum_r_from_other.extend(padding)
            
        feed[self.r_from_others] = sum_r_from_other

        # Run policy update
        _, meta_loss = sess.run(
            [self.policy_op_prime, self.meta_loss], feed_dict=feed)

        return meta_loss

class PolicyNew(object):
    """Helper class for policy updates."""
    
    def __init__(self, params, l_obs, l_action, agent_name):
        self.obs = tf.placeholder(tf.float32, [None, l_obs], 'obs_new')
        self.action_taken = tf.placeholder(
            tf.float32, [None, l_action], 'action_taken')
            
        prefix = agent_name + '/policy_main/policy/'
        with tf.variable_scope('policy_new'):
            h1 = tf.nn.relu(
                tf.nn.xw_plus_b(
                    self.obs, 
                    params[prefix + 'actor_h1/kernel:0'],
                    params[prefix + 'actor_h1/bias:0']))
            h2 = tf.nn.relu(
                tf.nn.xw_plus_b(
                    h1,
                    params[prefix + 'actor_h2/kernel:0'], 
                    params[prefix + 'actor_h2/bias:0']))
            out = tf.nn.xw_plus_b(
                h2,
                params[prefix + 'actor_out/kernel:0'],
                params[prefix + 'actor_out/bias:0'])
                
        self.probs = tf.nn.softmax(out)