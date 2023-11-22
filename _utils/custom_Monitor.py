import time
import warnings
from typing import Optional, Tuple, List

import numpy as np
import os
import torch as th


from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn, VecEnvWrapper

from stable_baselines3.common.logger import TensorBoardOutputFormat
class Custom_VecMonitor(VecEnvWrapper):
    """
    A vectorized monitor wrapper for *vectorized* Gym environments,
    it is used to record the episode reward, length, time and other data.

    Some environments like `openai/procgen <https://github.com/openai/procgen>`_
    or `gym3 <https://github.com/openai/gym3>`_ directly initialize the
    vectorized environments, without giving us a chance to use the ``Monitor``
    wrapper. So this class simply does the job of the ``Monitor`` wrapper on
    a vectorized level.

    :param venv: The vectorized environment
    :param filename: the location to save a log file, can be None for no log
    :param info_keywords: extra information to log, from the information return of env.step()
    """

    def __init__(
        self,
        venv: VecEnv,
        filename: Optional[str] = None,
        info_keywords: Tuple[str, ...] = (),
        obs_names: List[str] = None,
            report_maxes=True,
    ):
        # Avoid circular import
        from stable_baselines3.common.monitor import Monitor, ResultsWriter

        # This check is not valid for special `VecEnv`
        # like the ones created by Procgen, that does follow completely
        # the `VecEnv` interface
        try:
            is_wrapped_with_monitor = venv.env_is_wrapped(Monitor)[0]
        except AttributeError:
            is_wrapped_with_monitor = False

        if is_wrapped_with_monitor:
            warnings.warn(
                "The environment is already wrapped with a `Monitor` wrapper"
                "but you are wrapping it with a `VecMonitor` wrapper, the `Monitor` statistics will be"
                "overwritten by the `VecMonitor` ones.",
                UserWarning,
            )

        VecEnvWrapper.__init__(self, venv)
        self.episode_count = 0
        self.t_start = time.time()

        env_id = None
        if hasattr(venv, "spec") and venv.spec is not None:
            env_id = venv.spec.id

        self.results_writer: Optional[ResultsWriter] = None
        if filename:
            file = os.path.join(filename, 'custom_metrics')
            self.results_writer = TensorBoardOutputFormat(folder=file)

        self.obs_names = obs_names
        if 'actual_netload_now' in self.obs_names:
            self.community_netloads = []
            self.daily_community_netloads = [[]]*24

        if 'SoC_settle' in self.obs_names:
            self.daily_community_SoCs = [[]]*24

        self.report_maxes = report_maxes
        if self.report_maxes:
            self.max_return = -np.inf

        self.info_keywords = info_keywords
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)

    def reset(self) -> VecEnvObs:
            obs = self.venv.reset()
            self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
            self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)

            if 'actual_netload_now' in self.obs_names:
                self.community_netloads = []
                self.daily_community_netloads = [[]]*24

            if 'SoC_settle' in self.obs_names:
                self.daily_community_SoCs = [[]]*24

            return obs

    def step_wait(self) -> VecEnvStepReturn:
            obs, rewards, dones, infos = self.venv.step_wait()
            self.episode_returns += rewards
            self.episode_lengths += 1

            if 'actual_netload_now' in self.obs_names:
                netload_settle_step = obs[:,self.obs_names.index('actual_netload_now')]
                community_netload_step = np.sum(netload_settle_step)
                self.community_netloads.append(community_netload_step)


            new_infos = list(infos[:])
            if all(dones): #This assumes termination of all envs at the same time!!
                assert len(dones) == self.num_envs

                mean_returns = np.mean(self.episode_returns)
                self.results_writer.writer.add_histogram('custom_metrics/Building_returns', th.Tensor(self.episode_returns), self.episode_count)
                self.results_writer.writer.add_scalar('custom_metrics/Building_mean_returns', mean_returns, self.episode_count)

                for environment in range(self.num_envs):
                    self.results_writer.writer.add_scalar('custom_metrics/Building_' + str(environment + 1),self.episode_returns[environment], self.episode_count)


                # add netload metrics
                if 'actual_netload_now' in self.obs_names:
                    # ToDo: calculate average daily energy imported
                    avg_daily_energy_imported = np.mean(np.maximum(self.community_netloads, 0)) * 24
                    self.results_writer.writer.add_scalar('Power Quality/Average_daily_energy_imported', avg_daily_energy_imported, self.episode_count)

                    # ToDo: calculate average daily energy exported
                    avg_daily_energy_exported = -np.mean(np.minimum(self.community_netloads, 0)) * 24
                    self.results_writer.writer.add_scalar('Power Quality/Average_daily_energy_exported', avg_daily_energy_exported, self.episode_count)

                    # ToDo: calculate average daily peak load
                    # the daily peak load is the peak load of the community in a day, a day is defined as 24 hours
                    # first we calculate the peaks of each day, from the community netloads
                    daily_peak_loads = [np.max(self.community_netloads[i:i+24]) for i in range(0, len(self.community_netloads), 24)]
                    # then we calculate the average of the daily peaks
                    avg_daily_peak_load = np.mean(daily_peak_loads)
                    self.results_writer.writer.add_scalar('Power Quality/Average_daily_peak_load', avg_daily_peak_load, self.episode_count)

                    # ToDo: calculate average daily peak export
                    # the daily peak export is the peak export of the community in a day, a day is defined as 24 hours
                    # first we calculate the peaks of each day, from the community netloads
                    daily_peak_exports = [np.min(self.community_netloads[i:i+24]) for i in range(0, len(self.community_netloads), 24)]
                    # then we calculate the average of the daily peaks
                    avg_daily_peak_export = np.mean(daily_peak_exports)

                    # ToDo: calculate total peak load
                    total_peak_load = np.max(self.community_netloads)
                    self.results_writer.writer.add_scalar('Power Quality/Total_peak_load', total_peak_load, self.episode_count)

                    # ToDo: calculate total peak export
                    total_peak_export = -np.min(self.community_netloads)
                    self.results_writer.writer.add_scalar('Power Quality/Total_peak_export', total_peak_export, self.episode_count)

                    #ToDo: calculate average daily ramping rate
                    # the ramping rate is the delta between timesteps, so the average daily ramping rate is the average of the ramping rates of all timesteps in a day
                    ramping_rates = np.abs(np.diff(self.community_netloads))
                    avg_daily_ramping_rate = np.mean(ramping_rates) * 24
                    self.results_writer.writer.add_scalar('Power Quality/Average_daily_ramping_rate', avg_daily_ramping_rate, self.episode_count)

                    #ToDo: calculate daily load factor
                    # the load factor is the average load divided by the peak load
                    daily_mean_loads = [np.mean(self.community_netloads[i:i+24]) for i in range(0, len(self.community_netloads), 24)]
                    daily_load_factors = [1 - daily_mean_loads[i]/daily_peak_loads[i] for i in range(len(daily_mean_loads))]
                    avg_daily_load_factor = np.mean(daily_load_factors)
                    # we report the complement of the load factor, so that 0 is the best possible value
                    self.results_writer.writer.add_scalar('Power Quality/Average_daily_load_factor', avg_daily_load_factor, self.episode_count)

                    # ToDo: calculate monthly load factor
                    # the load factor is the average load divided by the peak load
                    monthly_mean_loads = [np.mean(self.community_netloads[i:i+24*30]) for i in range(0, len(self.community_netloads), 24*30)]
                    monthly_peaks = [np.max(self.community_netloads[i:i+24*30]) for i in range(0, len(self.community_netloads), 24*30)]
                    monthly_load_factors = [1 - monthly_mean_loads[i]/monthly_peaks[i] for i in range(len(monthly_mean_loads))]
                    avg_monthly_load_factor = np.mean(monthly_load_factors)
                    # we report the complement of the load factor, so that 0 is the best possible value
                    self.results_writer.writer.add_scalar('Power Quality/Average_monthly_load_factor', avg_monthly_load_factor, self.episode_count)

                    self.community_netloads = []
                    if self.report_maxes and self.max_return < mean_returns:
                        self.max_return = mean_returns
                        self.results_writer.writer.add_scalar('MaxSummary/Max_Return', self.max_return, self.episode_count)

                        # add the netload metrics for this run here
                        self.results_writer.writer.add_scalar('MaxSummary/Average_daily_energy_imported', avg_daily_energy_imported, self.episode_count)
                        self.results_writer.writer.add_scalar('MaxSummary/Average_daily_energy_exported', avg_daily_energy_exported, self.episode_count)
                        self.results_writer.writer.add_scalar('MaxSummary/Average_daily_peak_load', avg_daily_peak_load, self.episode_count)
                        self.results_writer.writer.add_scalar('MaxSummary/Average_daily_peak_export', avg_daily_peak_export, self.episode_count)
                        self.results_writer.writer.add_scalar('MaxSummary/Total_peak_load', total_peak_load, self.episode_count)
                        self.results_writer.writer.add_scalar('MaxSummary/Total_peak_export', total_peak_export, self.episode_count)
                        self.results_writer.writer.add_scalar('MaxSummary/Average_daily_ramping_rate', avg_daily_ramping_rate, self.episode_count)
                        self.results_writer.writer.add_scalar('MaxSummary/Average_daily_load_factor', avg_daily_load_factor, self.episode_count)
                        self.results_writer.writer.add_scalar('MaxSummary/Average_monthly_load_factor', avg_monthly_load_factor, self.episode_count)


                for i in range(len(dones)):
                    info = infos[i].copy()
                    episode_return = self.episode_returns[i]
                    episode_length = self.episode_lengths[i]
                    episode_info = {"r": episode_return,
                                    "l": episode_length,
                                    "t": round(time.time() - self.t_start, 6)
                                    }
                    for key in self.info_keywords:
                        episode_info[key] = info[key]
                    info["episode"] = episode_info
                    self.episode_count += 1
                    self.episode_returns[i] = 0
                    self.episode_lengths[i] = 0
                    new_infos[i] = info


            return obs, rewards, dones, new_infos

    def close(self) -> None:
            if self.results_writer:
                self.results_writer.close()
            return self.venv.close()