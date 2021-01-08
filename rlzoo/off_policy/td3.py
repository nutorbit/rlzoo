import tensorflow as tf

from tensorflow.keras import activations

from rlzoo.off_policy.ddpg import DDPG


class TD3(DDPG):
    """
    Twin Delayed Deep Deterministic Policy Gradient

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
        target_noise: noise in target network
        noise_clip: noise clip
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
                 output_activation=activations.linear,
                 target_noise=0.2,
                 noise_clip=0.5,
                 name="ddpg"):

        super().__init__(obs_dim, act_dim, gamma, lr, tau, interval_target, hiddens, activation, output_activation, name=name)
        self.target_noise = target_noise
        self.noise_clip = noise_clip

    @tf.function
    def critic_loss(self, batch):
        """
        L(s, a) = (y - Q(s,a))^2
        Where,
            y(s, a) = r(s, a) + (1 - done) * gamma * Q'(s', a'); a' ~ u'(s') + noise
        """

        # next action
        next_act = self.actor_target(batch['next_obs'])
        noise = self.get_noise(shape=next_act.shape)
        next_act = tf.clip_by_value(next_act + noise, -1, 1)

        q_target1, q_target2 = self.critic_target(batch['next_obs'], next_act)
        q_target = tf.minimum(q_target1, q_target2)
        y = batch['rew'] + (1 - batch['done']) * self.gamma * tf.stop_gradient(q_target)

        q1, q2 = self.critic(batch['obs'], batch['act'])

        loss1 = tf.reduce_mean(tf.square(y - q1))
        loss2 = tf.reduce_mean(tf.square(y - q2))

        return loss1 + loss2

    def get_noise(self, shape):
        """
        Random noise for exploration and prevent overestimate in Q
        Args:
            shape: shape of noise
        """

        noise = tf.random.normal(shape=shape)
        noise = tf.clip_by_value(noise, -self.noise_clip, self.noise_clip)
        return noise


if __name__ == "__main__":
    tmp = TD3(3, 3)
    batch = {
        "obs": tf.ones((10, 3)),
        "act": tf.ones((10, 3)),
        "next_obs": tf.ones((10, 3)),
        "rew": tf.ones((10, 1)),
        "done": tf.ones((10, 1))
    }
    print(tmp.update_params(batch, 0))
