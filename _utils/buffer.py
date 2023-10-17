import numpy as np
import scipy

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0,
         x1,
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def normalize_buffer_entry(buffer, key):
    array = []
    for episode in buffer.keys():
        episode_array = [step[key] for step in buffer[episode]]
        array.extend(episode_array)

    mean = np.mean(array)
    std = np.std(array)

    for episode in buffer.keys():
        for t in range(len(buffer[episode])):
            buffer[episode][t][key] = (buffer[episode][t][key] - mean) / (std + 1e-10)

    return buffer
class RecurrentExperienceReplay:
    def __init__(self, max_length=1e4, trajectory_length=1, action_types=None, multivariate=True):
        self.max_length = max_length #ToDo: rename to buffer lentgh
        self.buffer = {}  # a dict of lists, each entry is an episode which is itself a list of entries such as below
        self.last_episode = []
        self.action_types = action_types
        self.trajectory_length = trajectory_length  # trajectory length
        self.multivariate = multivariate


    def add_entry(self, actions_taken, log_probs, values, observations, rewards, critic_states=None, actor_states=None, episode=0):
        entry = {}
        if actions_taken is not None:
            entry['actions_taken'] = actions_taken
        if log_probs is not None:
            entry['log_probs'] = log_probs
        if values is not None:
            entry['values'] = values
        if observations is not None:
            entry['observations'] = observations
        if rewards is not None:
            entry['rewards'] = rewards
        if critic_states is not None:
            entry['critic_states'] = critic_states
        if actor_states is not None:
            entry['actor_states'] = actor_states

        if episode not in self.buffer: #ToDo: we might need to change this for asynch stuff
            self.buffer[episode] = []
        self.buffer[episode].append(entry)

    def clear_buffer(self):
        self.buffer = {}

    def generate_availale_indices(self):

        #get available indices
        available_indices = []
        for episode in self.buffer:
            for step in range(len(self.buffer[episode]) - self.trajectory_length):
                available_indices.append([episode, step])

        self.available_indices = available_indices
        return True

    def should_we_learn(self):
        buffer_length = 0
        for episode in self.buffer:
            buffer_length += len(self.buffer[episode])

        if buffer_length >= self.max_length:
            return True
        else:
            return False

    # This will need help
    def calculate_advantage(self, gamma=0.99, gae_lambda=0.95, normalize=True):

        self.buffer = normalize_buffer_entry(self.buffer, key='rewards')

        for episode in self.buffer:
            V_episode = [step['values'] for step in self.buffer[episode]]
            V_pseudo_terminal = V_episode[-1] #adding a pseudo bootstrap from our critic because we have long episodes

            r_episode = [step['rewards'] for step in self.buffer[episode]]
            r_episode.append(V_pseudo_terminal)
            r_episode_array = np.array(r_episode)

            G_episode = discount_cumsum(r_episode_array, gamma)[:-1] # removing the last one here because we added the value bootstrap
            for t in range(len(G_episode)):
                self.buffer[episode][t]['returns'] = G_episode[t]

        A = []
        for episode in self.buffer: #because we need to calculate those separately!
            V_episode = [step['values'] for step in self.buffer[episode]]
            V_pseudo_terminal = V_episode[-1]
            V_episode.append(V_pseudo_terminal)
            V_episode = np.array(V_episode)

            r_episode = [step['rewards'] for step in self.buffer[episode]]
            r_episode.append(V_pseudo_terminal)
            r_episode = np.array(r_episode)

            deltas = (r_episode[:-1] ) + gamma * V_episode[1:] - V_episode[:-1]
            A_eisode = discount_cumsum(deltas, gamma * gae_lambda)
            A_eisode = A_eisode.tolist()
            A.extend(A_eisode)
            for t in range(len(A_eisode)):
                self.buffer[episode][t]['advantages'] = A_eisode[t]

        #normalize advantage:
        if normalize:
            self.buffer = normalize_buffer_entry(self.buffer, key='advantages')
        #ToDo: do some research if normalizing rewards here is useful

        return True

    def _fetch_buffer_entry(self, batch_indices, key, subkeys=False, only_first_entry=False):
        #godl: trajectory_start:trajectory_start+self.trajectory_length
        # if subkeys: #for nested buffer entries
        #     fetched_entry = {}
        #     for subkey in subkeys:
        #         fetched_entry[subkey] = [self.buffer[sample_episode][trajectory_start][key][subkey]
        #                                       for [sample_episode, trajectory_start] in batch_indices]
        #
        # else:
        if self.trajectory_length <=1 or only_first_entry:
            fetched_entry = [self.buffer[sample_episode][trajectory_start][key]
                                          for [sample_episode, trajectory_start] in batch_indices]
        else:
            fetched_entry = []
            for [sample_episode, trajectory_start] in batch_indices:
                fetched_trajectory = [self.buffer[sample_episode][trajectory_start + step][key] for step in range(self.trajectory_length)]
                fetched_entry.append(fetched_trajectory)

        return fetched_entry

    #ToDo: add the reward offset piece here!
    def fetch_batch(self, batchsize=32, indices=None,
                    keys=['actions_taken', 'log_probs', 'observations', 'advantages', 'returns'],

                    ):

        np.random.shuffle(self.available_indices)
        batch_indices = self.available_indices[:batchsize]

        #ToDo: implement trajectories longer than 1, might be base on same code as DQN buffer

        batch = {}
        for key in keys:
            batch[key] = self._fetch_buffer_entry(batch_indices, key)

        return batch