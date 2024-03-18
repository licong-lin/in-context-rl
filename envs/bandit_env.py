import gym
import numpy as np
import torch
import pdb

from envs.base_env import BaseEnv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def sample(dim, act_num, H, var,  type='uniform'):  
    
    
    if type == 'uniform':
        means = np.random.uniform(0, 1, dim)
    elif type == 'bernoulli':
        means = np.random.beta(1, 1, dim)
    else:
        raise NotImplementedError
        
        
    
    if type=='uniform':
        action_set=np.random.uniform(low=-1,high=1,size=(act_num,dim))
    elif type == 'bernoulli':
        assert dim==act_num
        assert type=='bernoulli'
        action_set=np.eye(dim)
        
        
    
    
    #np.random.seed()
    env = BanditEnv(means,action_set,H, var=var, type=type)
    #pdb.set_trace()
    return env







# def sample(dim, act_num, H, var,  type='uniform'):           ##for Thomposon sampling
#     if type == 'uniform':
#         means = np.random.normal(loc=0, scale=1, size=(dim,))    ##prior_std=1.    
       
#     elif type == 'bernoulli':
#         means = np.random.beta(1, 1, dim)
#     else:
#         raise NotImplementedError
#     #np.random.seed(17)
#     action_set=np.random.uniform(low=-1,high=1,size=(act_num,dim))
    
    
#     #np.random.seed()
#     env = BanditEnv(means,action_set,H, var=var, type=type)     ##var is noise_var
#     return env




class BanditEnv(BaseEnv):
    def __init__(self, means,action_set, H, var=0.0, type='uniform'):
        self.means = means
        self.action_set=action_set
        
        opt_a_index = np.argmax(np.sum(means.reshape(1,-1)*action_set,axis=1))
        #opt_a_index=np.argmax(means)
        
        self.opt_a_index = opt_a_index
       
        
        
        self.dim = len(means)
        self.act_num = len(action_set)
        
        self.opt_a = np.zeros(self.act_num)
        self.opt_a[opt_a_index]=1
        
        #self.observation_space = gym.spaces.Box(low=1, high=1, shape=(1,))
        #self.action_space = gym.spaces.Box(low=0, high=1, shape=(self.dim,))
        
        self.state = np.array([1])
        self.var = var
        self.dx = 1
        self.du = self.act_num
        self.topk = False
        self.type = type

        # some naming issue here
        self.H_context = H
        self.H = 1

    def get_arm_value(self, u):
        return np.sum(self.means * self.action_set[np.argmax(u),:])

    def reset(self):
        self.current_step = 0
        return self.state

    def transit(self, x, u):
        v = np.sum(self.action_set[np.argmax(u),:]*self.means)
        if self.type == 'uniform':
            r = v + np.random.normal(0, self.var)
        elif self.type == 'bernoulli':
            r = np.random.binomial(n=1, p=v)
        else:
            raise NotImplementedError
        return self.state.copy(), r

    def step(self, action):     ##execute 1 step
        if self.current_step >= self.H:
            raise ValueError("Episode has already ended")

        _, r = self.transit(self.state, action)
        self.current_step += 1
        done = (self.current_step >= self.H)

        return self.state.copy(), r, done, {}

    def deploy_eval(self, ctrl):
        # No variance during evaluation
        tmp = self.var
        self.var = 0.0
        res = self.deploy(ctrl)
        self.var = tmp
        return res
    
    def deploy(self, ctrl, return_probs=False):
        ob = self.reset()
        obs = []
        acts = []
        next_obs = []
        rews = []
        probs = []
        done = False

        while not done:
            if return_probs:
                act, prob = ctrl.act(ob, return_probs=return_probs)
                probs.append(prob)
            else:
                act = ctrl.act(ob)
            acts.append(act)

            obs.append(ob)

            ob, rew, done, _ = self.step(act)

            rews.append(rew)
            next_obs.append(ob)

        obs = np.array(obs)
        acts = np.array(acts)
        next_obs = np.array(next_obs)
        rews = np.array(rews)
        probs = np.array(probs)
        
        if return_probs:
            return obs, acts, next_obs, rews, probs
        return obs, acts, next_obs, rews
    
    


class BanditEnvVec(BaseEnv):
    """
    Vectorized bandit environment.
    """
    def __init__(self, envs):
        self._envs = envs
        self._num_envs = len(envs)
        self.dx = envs[0].dx
        self.du = envs[0].du

    def reset(self):
        return [env.reset() for env in self._envs]

    def step(self, actions):
        next_obs, rews, dones = [], [], []
        for action, env in zip(actions, self._envs):
            next_ob, rew, done, _ = env.step(action)
            next_obs.append(next_ob)
            rews.append(rew)
            dones.append(done)
        return next_obs, rews, dones, {}

    @property
    def num_envs(self):
        return self._num_envs

    @property
    def envs(self):
        return self._envs

    def deploy_eval(self, ctrl):
        # No variance during evaluation
        tmp = [env.var for env in self._envs]
        for env in self._envs:
            env.var = 0.0
        res = self.deploy(ctrl)
        for env, var in zip(self._envs, tmp):
            env.var = var
        return res

    def deploy(self, ctrl):
        x = self.reset()
        xs = []
        xps = []
        us = []
        rs = []
        done = False

        while not done:
            u = ctrl.act_numpy_vec(x)

            xs.append(x)
            us.append(u)

            x, r, done, _ = self.step(u)
            done = all(done)

            rs.append(r)
            xps.append(x)

        xs = np.concatenate(xs)
        us = np.concatenate(us)
        xps = np.concatenate(xps)
        rs = np.concatenate(rs)
        return xs, us, xps, rs

    def get_arm_value(self, us):
        values = [np.sum(env.means * u) for env, u in zip(self._envs, us)]
        return np.array(values)

