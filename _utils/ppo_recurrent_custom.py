import sys
import time
from copy import deepcopy
from typing import Any, ClassVar, Dict, Optional, Type, TypeVar, Union

import numpy as np
import torch as th
from TREX_env._utils.custom_buffer import RecurrentRolloutBuffer
from gymnasium import spaces
from sb3_contrib.common.recurrent.buffers import RecurrentDictRolloutBuffer
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from sb3_contrib.ppo_recurrent.policies import CnnLstmPolicy, MlpLstmPolicy, MultiInputLstmPolicy
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn, obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv

SelfRecurrentPPO = TypeVar("SelfRecurrentPPO", bound="RecurrentPPO")


class RecurrentPPO(OnPolicyAlgorithm):
    """
    Proximal Policy Optimization algorithm (PPO) (clip version)
    with support for recurrent policies (LSTM).

    Based on the original Stable Baselines 3 implementation.

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param clip_range_vf: Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param normalize_advantage: Whether to normalize or not the advantage
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpLstmPolicy": MlpLstmPolicy,
        "CnnLstmPolicy": CnnLstmPolicy,
        "MultiInputLstmPolicy": MultiInputLstmPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[RecurrentActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 128,
        batch_size: Optional[int] = 128,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        recalculate_lstm_states: bool = False,
        rewards_shift: int = 0,
        self_bootstrap_dones: bool = True,
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl
        self._last_lstm_states = None

        self.recalculate_lstm_states = recalculate_lstm_states
        self.rewards_shift = rewards_shift
        if rewards_shift > 0:
            self.rewards_shift_fifo = [None]*rewards_shift
        self.self_bootstrap_dones = self_bootstrap_dones

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        buffer_cls = RecurrentDictRolloutBuffer if isinstance(self.observation_space, spaces.Dict) else RecurrentRolloutBuffer

        self.policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs,  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)

        # We assume that LSTM for the actor and the critic
        # have the same architecture
        lstm = self.policy.lstm_actor

        if not isinstance(self.policy, RecurrentActorCriticPolicy):
            raise ValueError("Policy must subclass RecurrentActorCriticPolicy")

        single_hidden_state_shape = (lstm.num_layers, self.n_envs, lstm.hidden_size)
        # hidden and cell states for actor and critic
        self._last_lstm_states = RNNStates(
            (
                th.zeros(single_hidden_state_shape, device=self.device),
                th.zeros(single_hidden_state_shape, device=self.device),
            ),
            (
                th.zeros(single_hidden_state_shape, device=self.device),
                th.zeros(single_hidden_state_shape, device=self.device),
            ),
        )

        if self.recalculate_lstm_states:
            self.partial_rollout_episode_buffer = {}#added to make sure we can recalculate the updated LSTM states after learning
            self.partial_rollout_episode_buffer['zero_lstm_states'] = RNNStates(
            (
                th.zeros(single_hidden_state_shape, device=self.device),
                th.zeros(single_hidden_state_shape, device=self.device),
            ),
            (
                th.zeros(single_hidden_state_shape, device=self.device),
                th.zeros(single_hidden_state_shape, device=self.device),
            ),
        )
            self.partial_rollout_episode_buffer['obs'] = []
            self.partial_rollout_episode_buffer['episode_starts'] = []
            self.partial_rollout_episode_buffer['rewards'] = []
    #         self.partial_rollout_episode_buffer['states'] = [] #used for debugging purposes

        hidden_state_buffer_shape = (self.n_steps, lstm.num_layers, self.n_envs, lstm.hidden_size)

        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            hidden_state_buffer_shape,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert isinstance(
            rollout_buffer, (RecurrentRolloutBuffer, RecurrentDictRolloutBuffer)
        ), f"{rollout_buffer} doesn't support recurrent policy"

        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        if self.rewards_shift > 0:
            self.rewards_shift_fifo = [None]*self.rewards_shift
            n_steps -= self.rewards_shift
            print('n steps', n_steps)
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        # ToDo: recalculate the new lstm_states based on a buffer of obs, etc so we make sure they are fresh!
        if self.recalculate_lstm_states and len(self.partial_rollout_episode_buffer['obs']) > 0:
            with th.no_grad():
                for step in range(len(self.partial_rollout_episode_buffer['obs'])):
                    _obs_tensor = self.partial_rollout_episode_buffer['obs'][step]
                    _episode_starts = self.partial_rollout_episode_buffer['episode_starts'][step]
                    lstm_input_states = self.partial_rollout_episode_buffer['zero_lstm_states'] if step == 0 else _lstm_output_states

                    _actions, _values, _log_probs, _lstm_output_states = self.policy.forward(_obs_tensor,
                                                                                      lstm_input_states,
                                                                                      _episode_starts)

                    if step >= len(self.partial_rollout_episode_buffer['obs']) - self.rewards_shift:
                        # add stuff to the buffer
                        step_dict = dict()

                        step_dict['last_obs'] = _obs_tensor.cpu().numpy()
                        step_dict['actions'] = _actions.cpu().numpy()
                        step_dict['log_probs'] = _log_probs
                        step_dict['rewards'] = deepcopy(self.partial_rollout_episode_buffer['rewards'][step])  # This is what we are shifting all other things for
                        step_dict['values'] = _values  # This needs to be shifted, too
                        step_dict['last_dones'] = _episode_starts.cpu().numpy()  # we have to shift this, too because this affects value calcs
                        step_dict['last_lstm_states'] = _lstm_output_states

                        self.rewards_shift_fifo.append(step_dict)
                        del self.rewards_shift_fifo[0]

                        n_steps += 1

                self._last_lstm_states = _lstm_output_states


        lstm_states = deepcopy(self._last_lstm_states)
        # self.partial_rollout_episode_buffer has obs, episode starts, etc as dict entries in list format

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                last_obs = deepcopy(self._last_obs)
                obs_tensor = obs_as_tensor(last_obs, self.device)
                last_starts = deepcopy(self._last_episode_starts)
                episode_starts = th.tensor(last_starts, dtype=th.float32, device=self.device) #Episode starts means that the LSTM state gets reset

                actions, values, log_probs, lstm_states = self.policy.forward(obs_tensor, lstm_states, episode_starts)
                # self.partial_rollout_episode_buffer['states'].append(lstm_states) #for debugging purposes only

            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            if self.recalculate_lstm_states:
                with th.no_grad():
                    self.partial_rollout_episode_buffer['obs'].append(obs_tensor.clone())
                    self.partial_rollout_episode_buffer['episode_starts'].append(episode_starts.clone())
                    self.partial_rollout_episode_buffer['rewards'].append(rewards)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done_ in enumerate(dones):
                if (
                    done_
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_lstm_state = (
                            lstm_states.vf[0][:, idx : idx + 1, :].contiguous(),
                            lstm_states.vf[1][:, idx : idx + 1, :].contiguous(),
                        )
                        # terminal_lstm_state = None
                        episode_starts = th.tensor([False], dtype=th.float32, device=self.device)
                        terminal_value = self.policy.predict_values(terminal_obs, terminal_lstm_state, episode_starts)[0]

                        rewards[idx] += self.gamma * terminal_value

            if self.rewards_shift > 0: #If we're shifting rewards, we're also shifting dones
                step_dict = dict()

                step_dict['last_obs'] = deepcopy(self._last_obs)
                step_dict['actions'] = deepcopy(actions)
                step_dict['log_probs'] = deepcopy(log_probs)
                step_dict['rewards'] = deepcopy(rewards) #This is what we are shifting all other things for
                step_dict['values'] = deepcopy(values) #This needs to be shifted, too
                step_dict['last_dones'] = deepcopy(self._last_episode_starts) #we have to shift this, too because this affects value calcs
                step_dict['last_lstm_states'] = deepcopy(self._last_lstm_states)

                self.rewards_shift_fifo.append(step_dict)

                if all(dones) and self.self_bootstrap_dones:
                    # we have already collected the value bootstrap for this. ALL that remains is adding it to the rewards
                    # since we shifted the rewards by reward_shift >= 1
                    # our current batched obs are at [0]
                    # so the bootstrap will be at [1]
                    with th.no_grad():
                        _obs = deepcopy(self.rewards_shift_fifo[1]['last_obs'])
                        _obs = obs_as_tensor(_obs, self.device)
                        _lstm_states = deepcopy(self.rewards_shift_fifo[1]['last_lstm_states'])
                        _last_dones = deepcopy(self.rewards_shift_fifo[1]['last_dones'])
                        _last_dones =  th.tensor(_last_dones, dtype=th.float32, device=self.device)
                        actions, terminal_value, log_probs, lstm_states = self.policy.forward(_obs, _lstm_states, _last_dones)
                        value_bootstrap_squeezed = np.squeeze( terminal_value.cpu().numpy())
                        rewards = self.rewards_shift_fifo[self.rewards_shift]['rewards']
                        self.rewards_shift_fifo[self.rewards_shift]['rewards'] = rewards + self.gamma * value_bootstrap_squeezed

                # ToDo: Something about how we're handling the episode end dones seems off
                if self.rewards_shift_fifo[0] is not None:
                    buffer_last_obs = th.Tensor(self.rewards_shift_fifo[0]['last_obs'])
                    buffer_actions = self.rewards_shift_fifo[0]['actions']
                    buffer_log_probs = self.rewards_shift_fifo[0]['log_probs']
                    last_lstm_states = self.rewards_shift_fifo[0]['last_lstm_states']
                    buffer_rewards = self.rewards_shift_fifo[self.rewards_shift]['rewards']
                    buffer_values = self.rewards_shift_fifo[0]['values']
                    last_dones = self.rewards_shift_fifo[0]['last_dones']

                    rollout_buffer.add(
                        buffer_last_obs,
                        buffer_actions,
                        buffer_rewards,
                        last_dones,
                        buffer_values,
                        buffer_log_probs,
                        lstm_states=last_lstm_states,
                    )

                del self.rewards_shift_fifo[0] #popping the 0th entry
            else:

                buffer_last_obs = self._last_obs
                buffer_actions = actions
                buffer_rewards = rewards
                buffer_values = values
                buffer_log_probs = log_probs
                last_dones = self._last_episode_starts
                last_lstm_states = self._last_lstm_states

                rollout_buffer.add(
                    buffer_last_obs,
                    buffer_actions,
                    buffer_rewards,
                    last_dones,
                    buffer_values,
                    buffer_log_probs,
                    lstm_states=last_lstm_states,
                )

            # # Original
            # rollout_buffer.add(
            #     self._last_obs,
            #     actions,
            #     rewards,
            #     self._last_episode_starts,
            #     values,
            #     log_probs,
            #     lstm_states=self._last_lstm_states,
            # )

            self._last_obs = new_obs
            self._last_episode_starts = dones
            self._last_lstm_states = lstm_states

            if self.recalculate_lstm_states and all(dones):
                print('resetting partial buffer') #ToDo: remove after some tests
                self.partial_rollout_episode_buffer['obs'] = []
                self.partial_rollout_episode_buffer['episode_starts'] = []
                self.partial_rollout_episode_buffer['rewards'] = []

            # whenever the env resets, we need to make sure to collect up to actual rewards first!
            if all(dones) and self.rewards_shift>0:
                print('resetting reward shift buffer because dones')
                self.rewards_shift_fifo = [None] * self.rewards_shift
                # reset the n_steps
                n_steps -= self.rewards_shift


        with th.no_grad():
            # Compute value for the last timestep
            episode_starts = th.tensor(dones, dtype=th.float32, device=self.device)
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device), lstm_states.vf, episode_starts)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size,):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Convert mask from float to bool
                mask = rollout_data.mask > 1e-8

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

    #ToDO: so it turns out that the size of the rollout seems to be
    # n_steps - 4
    # i could understand n_steps_2
    # i could understand batchsize, but its not batchsize
                burn_in = int(self.batch_size/2)
                learn_seq = int(self.batch_size - burn_in)
                mask = mask[burn_in:]
                with th.no_grad():
                    _obs = rollout_data.observations[:burn_in, :]
                    _lstm_states = rollout_data.lstm_states
                    _starts = rollout_data.episode_starts[:burn_in]
                    _actions, _values, _log_probs, _lstm_states = self.policy.forward(_obs, _lstm_states, _starts)

                learn_obs = rollout_data.observations[burn_in:, :]
                learn_actions = actions[burn_in:, :]
                learn_starts = rollout_data.episode_starts[burn_in:]
                values, log_prob, entropy = self.policy.evaluate_actions(
                    learn_obs,
                    learn_actions,
                    _lstm_states,
                    learn_starts,
                )

                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages[burn_in:]
                if self.normalize_advantage:
                    advantages = (advantages - advantages[mask].mean()) / (advantages[mask].std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                old_log_probs = rollout_data.old_log_prob[burn_in:]
                ratio = th.exp(log_prob - old_log_probs)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.mean(th.min(policy_loss_1, policy_loss_2)[mask])

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()[mask]).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                # Mask padded sequences
                targets = rollout_data.returns[burn_in:]
                value_loss = th.mean(((targets - values_pred) ** 2)[mask])

                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    clipped_log_probs = th.clamp(-log_prob[mask], -1e5, 1e5)
                    entropy_loss = -th.mean(clipped_log_probs)
                else:
                    entropy_loss = -th.mean(entropy[mask])

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - old_log_probs
                    approx_kl_div = th.mean(((th.exp(log_ratio) - 1) - log_ratio)[mask]).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def learn(
        self: SelfRecurrentPPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "RecurrentPPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfRecurrentPPO:
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
                fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(step=self.num_timesteps)

            self.train()

        callback.on_training_end()

        return self
