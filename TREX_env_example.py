from trexenv import TrexEnv
import multiprocessing as mp
import TREX_Core
from TREX_env._utils.trex_utils import prep_trex, run_subprocess


'''The goal of this piece of code is to show how to:
    - launch the TREX-core (our digityal twin)
    - launch the TREX-gym env - connect the env to the subprocess
    - do some basic interactions with the env
    '''


if __name__ == '__main__':
    config_name = 'GymIntegration_test' #'The name of you experiment's config'
    TREX_path = TREX_Core.__path__[0] ##ToDo: James - adjust this to whichever path is yours, TREX has to be set up as a package
    launch_list = prep_trex(config_name)

    pool_size = int(mp.cpu_count()/2) #Adjust based on needs
    pool = mp.Pool(processes=pool_size)
    trex_results = pool.map_async(run_subprocess, launch_list) #this launches the TREX-Core sim in a non-blocking fashion (so it runs in the background)
    pool.close()

    kwargs = {'action_space_type': 'discrete', #discrete or continuous
             'action_space_entries': 30} #if discrete, we need to know how many quantitizations we want between the min and max defined in the config file

    trex_env = TrexEnv(TREX_path=TREX_path, config_name=config_name, **kwargs)

    # getting some useful stuff from the environment
    agents_action_keys = trex_env.get_action_keys() #this is a list of the names for each agent's actions
    agents_action_spaces = trex_env.get_action_spaces() #this is a dict of the action spaces for each agent
    agents_obs_spaces = trex_env.get_obs_spaces() #this is a dict of the observation spaces for each agent
    episode_length = trex_env.episode_length  # this is the length of the episode, also defined in the config
    n_agents = trex_env.n_agents  # because agents are defined in the config

    obs = trex_env.reset() #this should print out a warning. The reset only resets stuff internally in the gym env, it does not reset the connected TREX-core sim. Steven should be on this but it's not high priority atm

    for i in range(int(episode_length)):
        #query the policy
        actions = trex_env.action_space.sample()  # getting some dummy actions for illustration. Accepted types are np one dimensional arrays, ints or floats
        actions = list(actions) #the actions have to be a list, not a tuple. This is a limitation of the gym env

        obs, reward, terminated, truncated, info = trex_env.step(actions)

        #Disclaimer: Rewards at the first 2 steps of an episode are nans, because the market settles for 1 step ahead.
        # t==0 we have nothing in the market and therefore no reward
        # t==1 the market is processing the settlements of the previous timestep, so we have no reward
        # t==2 the has processed the settlements and now we get reward
        # it might make learning faster to accomodate for this shift somehow, but it is not strictly necessary!

        print(reward)

    trex_env.close() #this will attempt to kill the subprocess and the TREX-Core sim
    pool.terminate() #this will close the pool of processes of the TREX-Core sim. It is nonetheless possible that the sim will not properly shut down. If this happens, you will have to kill the Python IDE instance manually :-(
    print('done')

