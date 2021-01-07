import tensorflow as tf

from tensorflow.keras import activations
from tensorflow.keras.optimizers import Adam

from rlzoo.common import make_mlp


class QNetwork(tf.Module):
    """
    Q-network for DQN

    Args:
        obs_dim: observation dimension
        act_dim: action dimension
        hiddens: number of unit for each layer
        activation: activation function for each hidden layer
        output_activation: activation function for last layer (output layer)
        name: name of this class
    """

    def __init__(self,
                obs_dim,
                act_dim,
                hiddens=[32, 32],
                activation=activations.tanh,
                output_activation=activations.linear,
                name="QNetwork"):

        super().__init__(name=name)

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.q1 = make_mlp([obs_dim] + hiddens + [act_dim], activation, output_activation)
        self.q2 = make_mlp([obs_dim] + hiddens + [act_dim], activation, output_activation)

    @tf.function
    def soft_update(self, other_network, tau):
        other_variables = other_network.trainable_variables
        current_variables = self.trainable_variables

        for (current_var, other_var) in zip(current_variables, other_variables):
            current_var.assign((1. - tau) * current_var + tau * other_var)

    @tf.function
    def hard_update(self, other_network):
        self.soft_update(other_network, tau=1.)

    @tf.function
    def __call__(self, obs):
        out_q1 = self.q1(obs)
        out_q2 = self.q2(obs)
        return tf.minimum(out_q1, out_q2)


class DQN(tf.Module):
    """
    Deep Q-learning Network

    Args:
        obs_dim: observation dimension
        act_dim: action dimension
        gamma: discount factor
        lr: learning rate
        tau: tau for update target network
        interval_target: number of steps for update target
        hiddens: number of unit for each layer
        activation: activation function for each hidden layer
        output_activation: activation function for last layer (output layer)
        name: name of this class
    """

    def __init__(self,
                 obs_dim,
                 act_dim,
                 gamma=0.98,
                 lr=1e-3,
                 tau=0.05,
                 interval_target=2,
                 hiddens=[32, 32],
                 activation=activations.tanh,
                 output_activation=activations.linear):

        super().__init__(name="DQN")
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.tau = tau
        self.gamma = gamma
        self.interval_target = interval_target

        self.q = QNetwork(obs_dim, act_dim, hiddens, activation, output_activation, name="q")
        self.q_target = QNetwork(obs_dim, act_dim, hiddens, activation, output_activation, name="q_target")
        self.q_target.hard_update(self.q)

        self.opt = Adam(lr=lr)

    @tf.function
    def loss_fn(self, batch):
        obs, act, next_obs, rew = batch["obs"], batch["act"], batch["next_obs"], batch["rew"]

        q_target = self.q_target(next_obs)

        q_target = tf.stop_gradient(rew + self.gamma * tf.reduce_max(q_target, axis=1, keepdims=True))

        idx = tf.expand_dims(tf.range(0, obs.shape[0]), axis=1)  # (N, 1)
        ind = tf.concat([idx, act], axis=1)
        q = tf.expand_dims(tf.gather_nd(self.q(obs), ind), 1)

        print(q.shape, q_target.shape)

        loss = tf.reduce_sum(tf.losses.MSE(q_target, q))
        return loss

    @tf.function
    def update_network(self, batch):
        with tf.GradientTape() as tape:
            loss = self.loss_fn(batch)
        grads = tape.gradient(loss, self.q.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.q.trainable_variables))
        return loss

    def update_params(self, batch, i):
        """
        Update parameters

        Args:
            batch: dictionary of (`obs`, `act`, `next_obs`, `rew`)
            i: timestep

        Returns:
            loss
        """
        loss = self.update_network(batch)

        if i % self.interval_target == 0:
            self.q_target.soft_update(self.q, self.tau)

        return loss

    def get_action(self, obs):
        """
        Get action

        Args:
            obs: observation

        Returns:
            action index
        """

        q = self.q(obs)
        idx = tf.argmax(q)
        return idx


if __name__ == "__main__":
    tmp = DQN(5, 20)
    batch = {
        "obs": tf.convert_to_tensor([[1., 1., 1., 1., 1.], [1., 1., 1., 1., 1.]]),
        "act": tf.convert_to_tensor([[1], [2]]),
        "next_obs": tf.convert_to_tensor([[2., 2., 1., 2., 2.], [2., 2., 2., 2., 2.]]),
        "rew": tf.convert_to_tensor([[10.], [0.]])
    }
    print(tmp.update_params(batch, 0))

