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
from stable_baselines3.common.vec_env import VecNormalize, VecMonitor, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from gymnasium.wrappers import FrameStack, NormalizeObservation
#ToDo: make sure all the shit in the env works before coming back here
# test if grid_equivalent is always worse (or at best equal) to market scenario
# test if the battery works for fucks sake?!
# test if the quantities line uo - battery charge and netloads, etc...

if "__main__" == __name__:  # this is needed to make sure the code is not executed when imported
    config_name = "GymIntegration_test"

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tboard_logdir = f"runs/{current_time}"

    trex_env = TrexEnv(config_name=config_name, action_space_type='continuous', action_space_entries=None)
    #trex_env = ss.frame_stack_v1(trex_env, 4)
    trex_env = ss.pettingzoo_env_to_vec_env_v1(trex_env)

    # trex_env = ss.flatten_v0(trex_env, )
    # trex_env = ss.normalize_obs_v0(trex_env, env_min=0, env_max=1)

    # trex_env = FrameStack(trex_env, num_stack=5)
    # trex_env = FlattenObservation(trex_env)
    trex_env = SB3VecEnvWrapper(trex_env)
    num_envs = trex_env.num_envs
    print('number of pseudo envs', num_envs)
    trex_env = VecNormalize(trex_env, norm_obs=True, norm_reward=False, clip_obs=np.inf, clip_reward=np.inf, gamma=0.99,
                 epsilon=1e-08)
    # trex_env = VecFrameStack(trex_env, n_stack=5)

    unnormalized_env = VecMonitor(trex_env, filename=tboard_logdir) #can add extra arguments to monitor in info keywords, look up https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/vec_env/vec_monitor.html
    final_env = VecNormalize(unnormalized_env, norm_obs=False, norm_reward=True, clip_obs=np.inf, clip_reward=np.inf, gamma=0.99, epsilon=1e-08)
    final_env = VecFrameStack(final_env, n_stack=12, channels_order='first')
    # trex_env = ss.concat_vec_envs_v1(trex_env, 1, base_class="stable_baselines3")


    # trex_env = ss.vector.markov_vector_wrapper.MarkovVectorEnv(trex_env)
    # get current time to add to tensoboard logdic

    #set up SAC

    policy_dict = dict(log_std_init=0.0,
                       # squash_output=True,
                       )
    # model = sb3.PPO('MlpPolicy',
    #                 final_env,
    #                 verbose=0,
    #                 tensorboard_log=tboard_logdir,
    #                 device="cuda",
    #                 n_epochs=40,
    #                 max_grad_norm=1.0,
    #                 # target_kl=0.04,
    #                 n_steps=int(2048/num_envs),
    #                 batch_size=64,
    #                 stats_window_size=1,
    #                 ent_coef=0.00,
    #                 policy_kwargs=policy_dict,
    #                 )
    model = sb3.SAC('MlpPolicy',
                    final_env,
                    verbose=0,
                    tensorboard_log=tboard_logdir,
                    device="cuda",

                    # replay_buffer_class=HerReplayBuffer,
                    # buffer_size=num_envs * 24 * 100,
                    # replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy='future'),
                    # batch_size=256,
                    learning_starts=24*100,
                    # train_freq=1,
                    # target_update_interval=1,
                    # ent_coef='auto',
                    # target_entropy='auto',
                    policy_kwargs=policy_dict,
                    stats_window_size=1,
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
    eval_callback = EvalCallback(eval_env=final_env,
                                 best_model_save_path="models/",
                                 log_path=tboard_logdir, n_eval_episodes=1,
                                 eval_freq= 10*24*100, deterministic=True, render=False)

    model.learn(total_timesteps=1e7,
                callback=eval_callback
                )
    #evaluate the model, add the reward values to the tensorboard log
    #reset the env first
    # obs = trex_env.reset()
    #evaluate the model
    # mean_reward, std_reward = evaluate_policy(model, trex_env, n_eval_episodes=trex_env.num_envs, deterministic=True)
    #add the reward values to the tensorboard log
    # model.logger.record("eval/mean_reward", mean_reward)
    # model.logger.record("eval/std_reward", std_reward)