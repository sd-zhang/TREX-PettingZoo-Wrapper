from trexenv import TrexEnv
import datetime
#ToDo: make sure all the shit in the env works before coming back here
# test if grid_equivalent is always worse (or at best equal) to market scenario
# test if the battery works for fucks sake?!
# test if the quantities line uo - battery charge and netloads, etc...
from TREX_env._utils.buffer import RecurrentExperienceReplay
from TREX_env._utils.models import build_actor_critic_models, sample_pi

import tensorflow as tf
from tensorflow import keras as k

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tboard_logdir = f"runs/{current_time}"
config_name = "GymIntegration_test"

buffer_length = 7*24*10 # (for 10 agents)
trajectory_length = 2*24
reward_shift = 2

def batch_for_vecenv(agents_dict):
    agents_batch = [agents_dict[agent] for agent in agents_dict]
    return agents_batch

def sample_trajectories_and_store():
    pass


if "__main__" == __name__:  # this is needed to make sure the code is not executed when imported

    #initializing the environment, setting some help variables
    trex_env = TrexEnv(config_name=config_name, action_space_type='continuous', action_space_entries=None)
    agent_names = trex_env.agents
    agent_rollout_episodes = [index for index, agent in enumerate(agent_names)]

    # initializing the Learner stuff
    # experience replay buffer
    rerp_buffer = RecurrentExperienceReplay(max_length=buffer_length, trajectory_length=trajectory_length)

    # assuming shared agents
    observation_space_shape = trex_env.observation_spaces[agent_names[0]].shape
    observation_space_shape = observation_space_shape[-1]
    action_space_shape = trex_env.action_spaces[agent_names[0]].shape #assumes action space shape length 1
    action_space_shape = action_space_shape[-1]
    hidden_actor = hidden_critic = [64, 64]
    actor_dict, critic_dict = build_actor_critic_models(num_inputs=observation_space_shape, #ToDo: change this to read the obs space,
                                                        hidden_actor=hidden_actor,
                                                        actor_type='GRU',
                                                        hidden_critic=hidden_critic,
                                                        critic_type='GRU',
                                                        num_actions=action_space_shape)

    ppo_actor = actor_dict['model']
    ppo_actor.compile(optimizer=k.optimizers.Adam(learning_rate=1e-3, ), ) #ToDo: make this a variable and find the right default

    ppo_actor_dist = actor_dict['distribution']
    ppo_actor_states_buffer = actor_dict['initial_states_dummy']
    for key in ppo_actor_states_buffer:
        ppo_actor_states_buffer[key] = ppo_actor_states_buffer[key]*len(agent_names) #assuming each agent as parallel actor

    ppo_critic = critic_dict['model']
    ppo_critic.compile(optimizer=k.optimizers.Adam(learning_rate=1e-3, ), ) #ToDo: make this a variable and a proper default
    ppo_critic_loss = k.losses.MeanSquaredError()

    ppo_critic_states_buffer = critic_dict['initial_states_dummy']
    for key in ppo_actor_states_buffer:
        shape = [len(agent_names), ppo_actor_states_buffer[key].shape[-1]]  # assuming each agent as parallel actor

        ppo_actor_states_buffer[key] = tf.broadcast_to(ppo_actor_states_buffer[key], shape=shape)

    # learning loop
    #reset env
    agents_obs_t, agents_infos = trex_env.reset()

    actor_inputs = dict()
    actor_inputs['observations'] = tf.convert_to_tensor(batch_for_vecenv(agents_obs_t))
    for key in ppo_actor_states_buffer:
        actor_inputs[key] = ppo_actor_states_buffer[key]
    actor_outs = ppo_actor(actor_inputs)
    for key in ppo_actor_states_buffer:
        ppo_actor_states_buffer[key] = actor_outs.pop(key) #save the updated states
    pi_t = actor_outs.pop('pi')
    action_scaled, log_prob, a_unscaled = sample_pi(pi_t, ppo_actor_dist, action_space_shape)

    ready_to_learn = False
    while not ready_to_learn:
        # get sample actions
        agents_actions = dict()
        for agent_index, agent_name in enumerate(agent_names):
            agent_action = a_unscaled[agent_index]
            agents_actions[agent_name] = agent_action

        agents_obs_t, agents_rewards_tm2, agents_terminateds_t, agents_truncateds_t,  agents_infos_t = trex_env.step(agents_actions)

        actor_inputs = dict()
        actor_inputs['observations'] = tf.convert_to_tensor(batch_for_vecenv(agents_obs_t))
        for key in ppo_actor_states_buffer:
            actor_inputs[key] = ppo_actor_states_buffer[key]
        actor_outs = ppo_actor(actor_inputs)
        for key in ppo_actor_states_buffer:
            ppo_actor_states_buffer[key] = actor_outs.pop(key)  # save the updated states
        pi_t = actor_outs.pop('pi')
        action_scaled, log_prob, a_unscaled = sample_pi(pi_t, ppo_actor_dist, action_space_shape)

        #ToDo: make the env return the settle offset, so we can do this dynamically
        # we'll wait for 2 steps so we can align the reward to the agent_obs
        # we have to make sure we do truncation properly!
        for agent_index, agent_name in enumerate(agent_names):
            run_index = agent_rollout_episodes[agent_index]

            rerp_buffer.add_entry(actions_taken=agents_actions[agent_name],
                                  log_probs=log_prob[agent_index],
                                  values=0, #ToDo: this is where we store the value of the agent during rollout
                                  observations=agents_obs_t[agent_name],
                                  rewards=agents_rewards_tm2[agent_name],
                                  actor_states=None, #ToDo: figure out how to query this!
                                  critic_states=None,
                                  episode=run_index,
                                  )

        ready_to_learn = rerp_buffer.should_we_learn()

    # collect a batch
    rerp_buffer.generate_availale_indices()
    rerp_buffer.calculate_advantage()
    batch = rerp_buffer.fetch_batch(batchsize=32,
                                    keys=['actions_taken', 'observations', 'rewards', 'values', 'advantages'])


    print('.')