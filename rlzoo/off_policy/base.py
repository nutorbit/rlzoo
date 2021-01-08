import tensorflow as tf


class BaseNetwork(tf.Module):
    """
    Base class for policy network and value network
    """

    def __init__(self, name=None):
        super().__init__(name=name)

    @tf.function
    def soft_update(self, other_network, tau):
        other_variables = other_network.trainable_variables
        current_variables = self.trainable_variables

        for (current_var, other_var) in zip(current_variables, other_variables):
            current_var.assign((1. - tau) * current_var + tau * other_var)

    @tf.function
    def hard_update(self, other_network):
        self.soft_update(other_network, tau=1.)
