from trexenv import TrexEnv
import numpy as np
import pettingzoo as pz
import stable_baselines3 as sb3
import supersuit as ss
from pettingzoo.test import parallel_api_test

if "__main__" == __name__:  # this is needed to make sure the code is not executed when imported
    config_name = "GymIntegration_test"

    kwargs = {'action_space_type': 'continuous',  # discrete or continuous
              'action_space_entries': None}  # if discrete, we need to know how many quantitizations we want between the min and max defined in the config file

    trex_env = TrexEnv(config_name=config_name, **kwargs)

    trex_env = ss.pettingzoo_env_to_vec_env_v1(trex_env)

    model = sb3.PPO(sb3.ppo.MlpPolicy,
        trex_env,
        n_steps=40,
        verbose=2,
        batch_size=64,
    )
    model.learn(total_timesteps=10000)