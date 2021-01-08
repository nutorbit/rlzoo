import tensorflow as tf

from tensorflow.keras import activations
from tensorflow.keras.optimizers import Adam

from rlzoo.common import make_mlp
from rlzoo.off_policy.base import BaseNetwork


class Critic(BaseNetwork):
    """
    Critic network

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
                 name="Critic"):

        super().__init__(name=name)

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.q1 = make_mlp([obs_dim + act_dim] + hiddens + [act_dim], activation, output_activation)
        self.q2 = make_mlp([obs_dim + act_dim] + hiddens + [act_dim], activation, output_activation)

    @tf.function
    def __call__(self, obs, act):
        concat = tf.concat([obs, act], axis=1)
        return self.q1(concat), self.q2(concat)


class Actor(BaseNetwork):
    """
    Actor network

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
                 name="Actor"):

        super().__init__(name=name)

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.pi = make_mlp([obs_dim] + hiddens + [act_dim], activation, output_activation)

    @tf.function
    def __call__(self, obs):
        out = self.pi(obs)
        return out


class DDPG(tf.Module):
    """
    Deep Deterministic Policy Gradient

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
                 output_activation=activations.linear,
                 name="ddpg"):

        super().__init__(name=name)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.tau = tau
        self.gamma = gamma
        self.interval_target = interval_target

        self.actor = Actor(obs_dim, act_dim, hiddens, activation, output_activation, name="actor")
        self.critic = Critic(obs_dim, act_dim, hiddens, activation, output_activation, name="critic")

        self.actor_target = Actor(obs_dim, act_dim, hiddens, activation, output_activation, name="actor_target")
        self.critic_target = Critic(obs_dim, act_dim, hiddens, activation, output_activation, name="critic_target")
        self.actor_target.hard_update(self.actor)
        self.critic_target.hard_update(self.critic)

        self.actor_opt = Adam(lr, name="actor_optimizer")
        self.critic_opt = Adam(lr, name="critic_optimizer")

    def update_params(self, batch, i):
        """
        Update parameters

        Args:
            batch: dictionary of (`obs`, `act`, `next_obs`, `rew`)
            i: timestep

        Returns:
            loss
        """

        critic_loss = self.update_critic(batch)
        actor_loss = self.update_actor(batch)

        if i % self.interval_target == 0:
            self.actor_target.soft_update(self.actor, self.tau)
            self.critic_target.soft_update(self.critic, self.tau)

        return {
            "critic_loss": critic_loss,
            "actor_loss": actor_loss
        }

    @tf.function
    def update_actor(self, batch):
        with tf.GradientTape() as tape:
            loss = self.actor_loss(batch)

        # Optimize the actor
        grads = tape.gradient(loss, self.actor.trainable_variables)
        self.actor_opt.apply_gradients(zip(grads, self.actor.trainable_variables))

        return loss

    @tf.function
    def update_critic(self, batch):
        with tf.GradientTape() as tape:
            loss = self.critic_loss(batch)

        # Optimize the critic
        grads = tape.gradient(loss, self.critic.trainable_variables)
        self.critic_opt.apply_gradients(zip(grads, self.critic.trainable_variables))

        return loss

    @tf.function
    def actor_loss(self, batch):
        """
        L(s) = -E[Q(s, a)| a~u(s)]
        """

        act = self.actor(batch['obs'])
        q1, q2 = self.critic(batch['obs'], act)
        loss = -tf.reduce_mean(q1)
        return loss

    @tf.function
    def critic_loss(self, batch):
        """
        L(s, a) = (y - Q(s,a))^2
        Where, y(s, a) = r(s, a) + (1 - done) * gamma * Q'(s', a'); a' ~ u'(s')
        """

        next_act = self.actor_target(batch['next_obs'])
        q_target1, q_target2 = self.critic_target(batch['next_obs'], next_act)
        q_target = tf.minimum(q_target1, q_target2)
        y = batch['rew'] + (1 - batch['done']) * self.gamma * tf.stop_gradient(q_target)

        q1, q2 = self.critic(batch['obs'], batch['act'])

        loss1 = tf.reduce_mean(tf.square(y - q1))
        loss2 = tf.reduce_mean(tf.square(y - q2))

        return loss1 + loss2

    def get_action(self, obs):
        """
        Get action

        Args:
            obs: observation

        Returns:
            action index
        """

        out = self.actor(obs)
        return out


if __name__ == "__main__":
    tmp = DDPG(3, 3)
    batch = {
        "obs": tf.ones((10, 3)),
        "act": tf.ones((10, 3)),
        "next_obs": tf.ones((10, 3)),
        "rew": tf.ones((10, 1)),
        "done": tf.ones((10, 1))
    }
    print(tmp.update_params(batch, 0))
