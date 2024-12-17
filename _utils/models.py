import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras as k


def build_multivar(concentrations, dist, actions):
    # independent_dists = []
    # for action in args:
    #     c0 = args[action]['c0']
    #     c1 = args[action]['c1']
    #     dist_action = dist(c0, c1)
    args = {}
    c0s, c1s  = tf.split(concentrations, 2, -1)
    c0s = tf.split(c0s, actions, -1)
    #c0s = tf.concat(c0s, axis=-2)
    c1s = tf.split(c1s, actions, -1)
    #c1s = tf.concat(c1s, axis=-2)
    args['concentration0'] = tf.concat(c0s, axis=-1)
    args['concentration1'] = tf.concat(c1s, axis=-1)
    betas = dist(**args)
    # sample = betas.sample(1)
    multivar_dist = tfp.distributions.Independent(betas, reinterpreted_batch_ndims=1)
    # sample_dis = multivar_dist.sample(1)
    return multivar_dist

def sample_pi(pi_params, actor_distrib, num_actions):
    dist = build_multivar(pi_params, actor_distrib, num_actions)

    a_dist = dist.sample()
    a_dist = tf.clip_by_value(a_dist, 1e-8, 0.999999)
    cardinality = len(a_dist.get_shape())
    to_be_reduced = [axis-1 for axis in range(cardinality, 1, -1)]
    a_dist = tf.squeeze(a_dist, axis=to_be_reduced)
    a_dist = a_dist.numpy().tolist()

    log_prob = dist.log_prob(a_dist)
    cardinality = len(log_prob.get_shape())
    to_be_reduced = [axis-1 for axis in range(cardinality, 1, -1)]
    log_prob = tf.squeeze(log_prob, axis=to_be_reduced)
    log_prob = log_prob.numpy().tolist()

    a_scaled = a_dist

    # #ToDO: redo this using the action space from the gym env
    # a_scaled = {}
    # keys = list(self.actions.keys())
    # for action_index in range(len(keys)):
    #     a = a_dist[action_index]
    #     min = self.actions[keys[action_index]]['min']
    #     max = self.actions[keys[action_index]]['max']
    #     a = min + (a * (max - min))
    #
    #     a_scaled[keys[action_index]] = a

    return a_scaled, log_prob, a_dist

def huber(x, epsilon=1e-10):
    x = tf.where(tf.math.greater(x, 1.0),
                             # essentially just huber function it so its bigger than 0
                             tf.abs(x),
                             tf.square(x))
    if epsilon > 0:
        x = tf.where(tf.math.greater(x, epsilon),
                     # essentially just huber function it so its bigger than 0
                     x,
                     epsilon)
    return x

def build_hidden_layer(signal, type='FFNN', num_hidden=32, name='Actor', initial_state=None, initializer=k.initializers.HeNormal()):

    if type == 'FFNN':
        signal = k.layers.Dense(num_hidden,
                                         activation="elu",
                                         kernel_initializer=initializer,
                                         name=name)(signal)
        return signal, None
    elif type == 'GRU':
        signal, last_state = k.layers.GRU(num_hidden,
                              activation='tanh',
                                recurrent_activation='sigmoid',
                              kernel_initializer=initializer,
                              return_sequences=True, return_state=True,
                              name=name)(signal, initial_state=initial_state)
        return signal, last_state

    else:
        print('requested layer type (', type, ') not recognized, failed to build ', name)
        return False, False

def build_hidden(internal_signal, inputs, outputs, hidden_neurons=[32,32,32], type='FFNN'):
    hidden_layer = 0
    initial_states_dummy = {}
    for num_hidden_neurons in hidden_neurons:
        if type == 'GRU':
            initial_state = k.layers.Input(shape=num_hidden_neurons, name='GRU_' + str(hidden_layer) + '_initial_state')
            inputs['GRU_'+str(hidden_layer)+'_state'] = initial_state
            initial_states_dummy['GRU_'+str(hidden_layer)+'_state'] = tf.zeros((1,num_hidden_neurons))

        else:
            initial_state = None
        internal_signal, last_state = build_hidden_layer(internal_signal,
                                       type=type,
                                       num_hidden=num_hidden_neurons,
                                       initial_state=initial_state,
                                       name='Actor_hidden_' + str(hidden_layer))
        if type == 'GRU':
            outputs['GRU_'+str(hidden_layer)+'_state'] = last_state
        hidden_layer += 1

    return internal_signal, inputs, outputs, initial_states_dummy

def build_actor(num_inputs=4, num_actions=3, hidden_actor=[32, 32, 32], actor_type='FFNN'):
    initializer = k.initializers.Orthogonal()
    inputs = {}
    outputs = {}

    shape = (num_inputs,) if actor_type != 'GRU' else (None, num_inputs,)
    internal_signal = k.layers.Input(shape=shape, name='Actor_Input')
    inputs['observations'] = internal_signal

    internal_signal,  inputs, outputs, initial_states_dummy = build_hidden(internal_signal,
                                                                           inputs=inputs,
                                                                           outputs=outputs,
                                                                           hidden_neurons=hidden_actor,
                                                                           type=actor_type)

    concentrations = k.layers.Dense(2 * num_actions,
                                    activation='tanh', #ToDo: test tanh vs None
                                    kernel_initializer=initializer,
                                    name='concentrations')(internal_signal)
    concentrations = huber(concentrations)
    outputs['pi'] = concentrations
    actor_model = k.Model(inputs=inputs, outputs=outputs)

    actor_distrib = tfp.distributions.Beta

    out_dict={'model': actor_model,
              'distribution': actor_distrib,
              'initial_states_dummy': initial_states_dummy}

    return out_dict

def build_critic(num_inputs=4, hidden_critic=[32, 32, 32], critic_type='FFNN'):
    initializer = k.initializers.Orthogonal()
    inputs = {}
    outputs = {}
    shape = (num_inputs,) if critic_type != 'GRU' else (None, num_inputs,)
    internal_signal = k.layers.Input(shape=shape, name='Critic_Input')
    inputs['observations'] = internal_signal

    internal_signal, inputs, outputs, initial_states_dummy = build_hidden(internal_signal,
                                                                          inputs=inputs,
                                                                          outputs=outputs,
                                                                          hidden_neurons=hidden_critic,
                                                                          type=critic_type)

    value = k.layers.Dense(1,
                           activation='tanh', #ToDo: test tanh vs None
                           kernel_initializer=initializer,
                           name='ValueHead')(internal_signal)
    outputs['value'] = value
    critic_model = k.Model(inputs=inputs, outputs=outputs)

    critic_dict = {'model': critic_model,
                   'initial_states_dummy': initial_states_dummy}
    return critic_dict

def build_actor_critic_models(num_inputs=4,
                              hidden_actor=[32, 32, 32],
                              actor_type='FFNN', #['FFNN', 'GRU'] #ToDo
                              hidden_critic=[32,32,32],
                              critic_type='FFNN', #['FFNN', 'GRU'] #ToDo
                              num_actions=4):
    # needs to return a suitable actor ANN, ctor PDF function and critic ANN
    initializer = tf.keras.initializers.HeNormal()
    actor_dict = build_actor(num_inputs=num_inputs,
                              num_actions=num_actions,
                              hidden_actor=hidden_actor,
                              actor_type=actor_type)
    critic_dict = build_critic(num_inputs=num_inputs,
                              hidden_critic=hidden_critic,
                              critic_type=critic_type)

    return actor_dict, critic_dict