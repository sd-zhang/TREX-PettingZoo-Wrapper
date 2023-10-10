from trexenv import TrexEnv
import numpy as np
import pettingzoo as pz
import stable_baselines3 as sb3
import supersuit as ss
from supersuit.vector.sb3_vector_wrapper import SB3VecEnvWrapper
from pettingzoo.test import parallel_api_test
import datetime
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecNormalize, VecMonitor, StackedObservations
from stable_baselines3.common.monitor import Monitor
#ToDo: make sure all the shit in the env works before coming back here
# test if grid_equivalent is always worse (or at best equal) to market scenario
# test if the battery works for fucks sake?!
# test if the quantities line uo - battery charge and netloads, etc...

if "__main__" == __name__:  # this is needed to make sure the code is not executed when imported
    config_name = "GymIntegration_test"

    kwargs = {'action_space_type': 'continuous',  # discrete or continuous
              'action_space_entries': None}  # if discrete, we need to know how many quantitizations we want between the min and max defined in the config file

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tboard_logdir = f"runs/{current_time}"

    trex_env = TrexEnv(config_name=config_name, **kwargs)
    trex_env = ss.pettingzoo_env_to_vec_env_v1(trex_env)
    num_envs = trex_env.num_envs
    trex_env = SB3VecEnvWrapper(trex_env)
    # trex_env = StackedObservations(trex_env, num_envs=num_envs,
    #             n_stack=12, channels_order='first', observation_space=trex_env.observation_space)
    # add the monitor wrapper
    trex_env = VecMonitor(trex_env, filename=tboard_logdir, info_keywords=('max_action', 'min_action')) #can add extra arguments to monitor in info keywords, look up https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/vec_env/vec_monitor.html
    trex_env = VecNormalize(trex_env, norm_obs=True, norm_reward=True, clip_obs=10., )

    # trex_env = ss.concat_vec_envs_v1(trex_env, 1, base_class="stable_baselines3")


    # trex_env = ss.vector.markov_vector_wrapper.MarkovVectorEnv(trex_env)
    # get current time to add to tensoboard logdic

    #set up SAC

    # train PPO
    # model = sb3.PPO('MlpPolicy',
    #                 trex_env,
    #                 verbose=1,
    #                 tensorboard_log=tboard_logdir,
    #                 device="cuda",
    #                 target_kl=0.04,
    #                 n_steps=128,
    #                 stats_window_size=2, #trex_env.num_envs,
    #                 ent_coef=0.01,
    #                 policy_kwargs=(dict(squash_output=True)),
    #                 batch_size=64)
    policy_dict = dict(# squash_output=True,
                       # use_sde=True,
                        #use_expln=True,
                       #log_std_init=0
                        )
    model = sb3.SAC('MlpPolicy',
                     trex_env,
                     verbose=0,
                     tensorboard_log=f"runs/{current_time}",
                     device="cuda",
                     ent_coef='auto',
                     learning_starts=1000,
                     batch_size=128,
                     policy_kwargs=policy_dict,
                     buffer_size=100000,
                     )
    # model = RecurrentPPO('MlpLstmPolicy',
    #                      trex_env,
    #                      verbose=0,
    #                      use_sde=True,
    #                      tensorboard_log=tboard_logdir,
          #                      device="cuda",
    #                      n_epochs=40,
    #                      target_kl=0.04,
    #                      n_steps=trex_env.num_envs * 128,
    #                      stats_window_size=1,
    #                      ent_coef=0.00,
    #                      policy_kwargs=policy_dict,
    #                      batch_size=128)
    # eval_callback = EvalCallback(eval_env=trex_env,
    #                              best_model_save_path="models/",
    #                              log_path=tboard_logdir, n_eval_episodes=1,
    #                              eval_freq= 5*24*100, deterministic=True, render=False)

    model.learn(total_timesteps=1e7,
                # callback=eval_callback
                )
    #evaluate the model, add the reward values to the tensorboard log
    #reset the env first
    # obs = trex_env.reset()
    #evaluate the model
    # mean_reward, std_reward = evaluate_policy(model, trex_env, n_eval_episodes=trex_env.num_envs, deterministic=True)
    #add the reward values to the tensorboard log
    # model.logger.record("eval/mean_reward", mean_reward)
    # model.logger.record("eval/std_reward", std_reward)