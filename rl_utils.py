from tqdm import tqdm
import numpy as np
import torch
import collections
import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity) 

    def add(self, state, action, reward, next_state, done): 
        self.buffer.append((state, action, reward, next_state, done)) 

    def sample(self, batch_size): 
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done 

    def size(self): 
        return len(self.buffer)

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))  #这一行代码计算了累积和。np.insert(a, 0, 0)在数组a的开头插入了一个零，然后np.cumsum()对插入零后的数组进行累积求和操作，得到一个新的数组cumulative_sum，其长度比原数组a多1
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size   #这一行代码计算了移动平均的中间部分。通过取累积和数组的切片操作，得到窗口大小之后的部分与窗口大小之前的部分的差值，然后除以窗口大小，得到中间部分的移动平均值。
    r = np.arange(1, window_size-1, 2)  #这一行代码生成了一个等差数组，表示移动平均的起始和结束阶段的权重。
    begin = np.cumsum(a[:window_size-1])[::2] / r   #这一行代码计算了移动平均的起始部分。通过取原始数据数组的切片操作，得到窗口大小之前的部分，然后进行累积求和，并与权重数组进行除法操作，得到起始部分的移动平均值。
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]   #这一行代码计算了移动平均的结束部分。通过取原始数据数组的逆向切片操作，得到窗口大小之后的部分，然后进行累积求和，并与权重数组进行除法操作，最后再反转数组的顺序，得到结束部分的移动平均值。
    return np.concatenate((begin, middle, end)) #这一行代码将起始部分、中间部分和结束部分的移动平均值进行拼接，得到最终的移动平均值数组，并将其作为函数的返回值。

def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list

def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)
                