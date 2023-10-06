from trexenv import TrexEnv
import numpy as np
import pettingzoo as pz
import stable_baselines3 as sb3
import supersuit as ss
from supersuit.vector.sb3_vector_wrapper import SB3VecEnvWrapper
from pettingzoo.test import parallel_api_test

if "__main__" == __name__:  # this is needed to make sure the code is not executed when imported
    config_name = "GymIntegration_test"

    kwargs = {'action_space_type': 'continuous',  # discrete or continuous
              'action_space_entries': None}  # if discrete, we need to know how many quantitizations we want between the min and max defined in the config file

    trex_env = TrexEnv(config_name=config_name, **kwargs)

    trex_env = ss.pettingzoo_env_to_vec_env_v1(trex_env)
    trex_env = SB3VecEnvWrapper(trex_env)
    # trex_env = ss.concat_vec_envs_v1(trex_env, 1, base_class="stable_baselines3")


    # trex_env = ss.vector.markov_vector_wrapper.MarkovVectorEnv(trex_env)

    model = sb3.ppo.PPO("MlpPolicy",
        trex_env,
        device="gpu",
        n_steps=8,
        verbose=1,
    )
    model.learn(total_timesteps=10000)