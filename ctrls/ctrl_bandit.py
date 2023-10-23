import itertools

import numpy as np
import scipy
import torch
from IPython import embed

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Controller:
    def __init__(self):
        self.name = None
        
    def set_batch(self, batch):
        self.batch = batch

    def set_batch_numpy_vec(self, batch):
        self.set_batch(batch)

    def set_env(self, env):
        self.env = env


class OptPolicy(Controller):
    def __init__(self, env, batch_size=1):
        super().__init__()
        self.env = env
        self.batch_size = batch_size

    def reset(self):
        return

    def act(self, x):
        return self.env.opt_a


    def act_numpy_vec(self, x):
        opt_as = [ env.opt_a for env in self.env ]
        return np.stack(opt_as, axis=0)
        # return np.tile(self.env.opt_a, (self.batch_size, 1))
        
        
        

        

class OffPolicy(Controller):
    def __init__(self, env, prob,batch_size=1):
        super().__init__()
        self.env = env
        self.prob = prob
        self.batch_size = batch_size

    def reset(self):
        return

    def act(self, x):
        rand_index=np.random.choice(self.env.act_num,p=self.prob)
        self.a=np.zeros(self.env.act_num)
        self.a[rand_index]=1.
        return self.a


    def act_numpy_vec(self, x):
        rand_indices = [ np.random.choice(env.act_num,p=self.prob) for env in self.env ]
        rand_indices=np.array(rand_indices).reshape(-1)
        for env in self.env:
            self.a=np.zeros((len(self.env),env.act_num))
            break
            
        for (i,idx) in enumerate(rand_indices):
            self.a[i,idx]=1.
        return self.a
        # return np.tile(self.env.opt_a, (self.batch_size, 1))


class GreedyOptPolicy(Controller):
    def __init__(self, env):
        super().__init__()
        self.env = env

    def reset(self):
        return

    def act(self, x):
        rewards = self.batch['context_rewards'].cpu().detach().numpy().flatten()
        if len(rewards)==0:
            i=np.random.choice(self.env.act_num)
            a=np.zeros(self.env.act_num)
            a[i]=1.
            self.a = a
            return self.a
        
        i = np.argmax(rewards)
        a = self.batch['context_actions'].cpu().detach().numpy()[0][i]
        self.a = a
        return self.a


class EmpMeanPolicy(Controller):
    def __init__(self, env, online=False, batch_size = 1):
        super().__init__()
        self.env = env
        self.online = online
        self.batch_size = batch_size

    def reset(self):
        return

    def act(self, x):
        actions = self.batch['context_actions'].cpu().detach().numpy()[0]
        rewards = self.batch['context_rewards'].cpu().detach().numpy().flatten()

        b = np.zeros(self.env.act_num)
        counts = np.zeros(self.env.act_num)
        for i in range(len(actions)):
            c = np.argmax(actions[i])
            b[c] += rewards[i]
            counts[c] += 1

        b_mean = b / np.maximum(1, counts)

        i = np.argmax(b_mean)
        j = np.argmin(counts)
        if self.online and counts[j] == 0:
            i = j
        
        a = np.zeros(self.env.act_num)
        a[i] = 1.0

        self.a = a
        return self.a

    def act_numpy_vec(self, x):
        actions = self.batch['context_actions']
        rewards = self.batch['context_rewards']

        b = np.zeros((self.batch_size, self.env.act_num))
        counts = np.zeros((self.batch_size, self.env.act_num))
        action_indices = np.argmax(actions, axis=-1)
        for idx in range(self.batch_size):
            actions_idx = action_indices[idx]
            rewards_idx = rewards[idx]
            for c in range(self.env.act_num):
                arm_rewards = rewards_idx[actions_idx == c]
                b[idx, c] = np.sum(arm_rewards)
                counts[idx, c] = len(arm_rewards)

        b_mean = b / np.maximum(1, counts)

        i = np.argmax(b_mean, axis=-1)
        j = np.argmin(counts, axis=-1)
        if self.online:
            mask = (counts[np.arange(self.batch_size), j] == 0)
            i[mask] = j[mask]

        a = np.zeros((self.batch_size, self.env.act_num))
        a[np.arange(self.batch_size), i] = 1.0

        self.a = a
        return self.a



class ThompsonSamplingPolicy(Controller):
    def __init__(self, env, std=1,  prior_std=1, batch_size=1):
        super().__init__()
        self.env = env
        self.variance = std**2
        
        
    
        self.prior_variance = prior_std**2
        
        self.normalize_lam=self.variance/self.prior_variance
        self.batch_size = batch_size
        
        self.t = 0
       
        self.theta = np.zeros(self.env.dim)
        self.cov = self.normalize_lam * np.eye(self.env.dim)



     
        
    def compare_actions(self,mean,cov):
        w=np.random.multivariate_normal(mean, cov, size=1)
        w=w.reshape(-1,1)
        index=np.argmax((self.env.action_set@w).reshape(-1))
        return index
    
    def compare_actions_bernoulli(self,successes,failures):
        sampled_theta = np.random.beta(1 + successes, 1 + failures)
        return np.argmax(sampled_theta)
        
        
        
        
    def act(self, x):
        
        hot_vector = np.zeros(self.env.act_num)
        
        
        if self.env.type=='bernoulli':
            if len(self.batch['context_rewards'][0]) < 1:
            
                index =self.compare_actions_bernoulli(np.zeros(self.env.act_num),np.zeros(self.env.act_num))
                hot_vector[index] = 1
                self.t += 1
            else:
                actions = self.batch['context_actions'].cpu().detach().numpy()[0]
                rewards = self.batch['context_rewards'].cpu().detach().numpy().flatten()

                actions_index=np.argmax(actions,axis=1)


                action_values=self.env.action_set[actions_index,:]
                
                successes = np.sum(action_values*rewards.reshape(-1,1),axis=0)
                failures = np.sum(action_values,axis=0)-successes
                index = self.compare_actions_bernoulli(successes,failures)
               
                
                hot_vector[index] = 1.
            return hot_vector


                
            
        
        
        
        
        
        
       
        
        if len(self.batch['context_rewards'][0]) < 1:
            
            index =self.compare_actions(self.theta,self.variance*np.linalg.inv(self.cov))
            hot_vector[index] = 1
            self.t += 1
          
        else:
            actions = self.batch['context_actions'].cpu().detach().numpy()[0]
            rewards = self.batch['context_rewards'].cpu().detach().numpy().flatten()
            
            actions_index=np.argmax(actions,axis=1)
          
            
            action_values=self.env.action_set[actions_index,:]

            cov = self.cov + action_values.T @ action_values
            cov_inv = np.linalg.inv(cov)

            theta = cov_inv @ action_values.T @ rewards.reshape(-1,1)
            
            hot_vector = np.zeros(self.env.act_num)
            index =self.compare_actions(theta.reshape(-1),self.variance*cov_inv)
            hot_vector[index] = 1.
          
            
        return hot_vector
            
            

            

            
            
    
    
    
    
    
    

class ThompsonSamplingPolicyAvg(Controller):
    def __init__(self, env, std=1,  prior_std=1, batch_size=1):
        super().__init__()
        self.env = env
        self.variance = std**2
        
        
    
        self.prior_variance = prior_std**2
        
        self.normalize_lam=self.variance/self.prior_variance
        self.batch_size = batch_size
        
        self.t = 0
       
        self.theta = np.zeros(self.env.dim)
        self.cov = self.normalize_lam * np.eye(self.env.dim)



     
        
    def compare_actions(self,mean,cov):
        w=np.random.multivariate_normal(mean, cov, size=1)
        w=w.reshape(-1,1)
        
        value_vec=(self.env.action_set@mean).reshape(-1)
        
        max_value = np.max(value_vec)
        
        index_list = np.where(value_vec==max_value)[0]
        
        index = np.random.choice(index_list)
        
        return index
    
    def compare_actions_bernoulli(self,successes,failures):
        value_vec = (1+successes)/(1+failures)
        
        
        max_value = np.max(value_vec)
        
        index_list = np.where(value_vec==max_value)[0]
        
        index = np.random.choice(index_list)
        
        return index
        
        
        
        
        
        
        
        
    def act(self, x):
        
        hot_vector = np.zeros(self.env.act_num)
        
        
        if self.env.type=='bernoulli':
            if len(self.batch['context_rewards'][0]) < 1:
            
                index =self.compare_actions_bernoulli(np.zeros(self.env.act_num),np.zeros(self.env.act_num))
                hot_vector[index] = 1
                self.t += 1
            else:
                actions = self.batch['context_actions'].cpu().detach().numpy()[0]
                rewards = self.batch['context_rewards'].cpu().detach().numpy().flatten()

                actions_index=np.argmax(actions,axis=1)


                action_values=self.env.action_set[actions_index,:]
                
                successes = np.sum(action_values*rewards.reshape(-1,1),axis=0)
                failures = np.sum(action_values,axis=0)-successes
                index = self.compare_actions_bernoulli(successes,failures)
                
                hot_vector[index] = 1.
            return hot_vector


                
            
        
        
        
        
        
        
       
        
        if len(self.batch['context_rewards'][0]) < 1:
            
            index =self.compare_actions(self.theta,self.variance*np.linalg.inv(self.cov))
            hot_vector[index] = 1
            self.t += 1
          
        else:
            actions = self.batch['context_actions'].cpu().detach().numpy()[0]
            rewards = self.batch['context_rewards'].cpu().detach().numpy().flatten()
            
            actions_index=np.argmax(actions,axis=1)
          
            
            action_values=self.env.action_set[actions_index,:]

            cov = self.cov + action_values.T @ action_values
            cov_inv = np.linalg.inv(cov)

            theta = cov_inv @ action_values.T @ rewards.reshape(-1,1)
            
            hot_vector = np.zeros(self.env.act_num)
            index =self.compare_actions(theta.reshape(-1),self.variance*cov_inv)
            hot_vector[index] = 1.
          
            
        return hot_vector
            
       
    
    
    
    
    
    
    
    
    
    
    
    
    
    
            
            
            
            
            
            
            
            

class PessMeanPolicy(Controller):
    def __init__(self, env, const=1.0, batch_size=1):
        super().__init__()
        self.env = env
        self.const = const
        self.batch_size = batch_size

    def reset(self):
        return

    def act(self, x):
        actions = self.batch['context_actions'].cpu().detach().numpy()[0]
        rewards = self.batch['context_rewards'].cpu().detach().numpy().flatten()

        b = np.zeros(self.env.dim)
        counts = np.zeros(self.env.dim)
        for i in range(len(actions)):
            c = np.argmax(actions[i])
            b[c] += rewards[i]
            counts[c] += 1

        b_mean = b / np.maximum(1, counts)

        # compute the square root of the counts but clip so it's at least one
        pens = self.const / np.maximum(1, np.sqrt(counts))
        bounds = b_mean - pens

        i = np.argmax(bounds)
        a = np.zeros(self.env.dim)
        a[i] = 1.0
        self.a = a
        return self.a


    def act_numpy_vec(self, x):
        actions = self.batch['context_actions']
        rewards = self.batch['context_rewards']

        b = np.zeros((self.batch_size, self.env.dim))
        counts = np.zeros((self.batch_size, self.env.dim))
        action_indices = np.argmax(actions, axis=-1)
        for idx in range(self.batch_size):
            actions_idx = action_indices[idx]
            rewards_idx = rewards[idx]
            for c in range(self.env.dim):
                arm_rewards = rewards_idx[actions_idx == c]
                b[idx, c] = np.sum(arm_rewards)
                counts[idx, c] = len(arm_rewards)

        b_mean = b / np.maximum(1, counts)

        # compute the square root of the counts but clip so it's at least one
        bons = self.const / np.maximum(1, np.sqrt(counts))
        bounds = b_mean - bons

        i = np.argmax(bounds, axis=-1)
        a = np.zeros((self.batch_size, self.env.dim))
        a[np.arange(self.batch_size), i] = 1.0
        self.a = a
        return self.a



class UCBPolicy(Controller):
    def __init__(self, env, const=1.0, batch_size=1):
        super().__init__()
        self.env = env
        self.const = const
        self.batch_size = batch_size

    def reset(self):
        return

    def act(self, x):
        actions = self.batch['context_actions'].cpu().detach().numpy()[0]
        rewards = self.batch['context_rewards'].cpu().detach().numpy().flatten()

        b = np.zeros(self.env.dim)
        counts = np.zeros(self.env.dim)
        for i in range(len(actions)):
            c = np.argmax(actions[i])
            b[c] += rewards[i]
            counts[c] += 1

        b_mean = b / np.maximum(1, counts)

        # compute the square root of the counts but clip so it's at least one
        bons = self.const / np.maximum(1, np.sqrt(counts))
        bounds = b_mean + bons

        i = np.argmax(bounds)
        a = np.zeros(self.env.dim)
        a[i] = 1.0
        self.a = a
        return self.a

    def act_numpy_vec(self, x):
        actions = self.batch['context_actions']
        rewards = self.batch['context_rewards']

        b = np.zeros((self.batch_size, self.env.dim))
        counts = np.zeros((self.batch_size, self.env.dim))
        action_indices = np.argmax(actions, axis=-1)
        for idx in range(self.batch_size):
            actions_idx = action_indices[idx]
            rewards_idx = rewards[idx]
            for c in range(self.env.dim):
                arm_rewards = rewards_idx[actions_idx == c]
                b[idx, c] = np.sum(arm_rewards)
                counts[idx, c] = len(arm_rewards)

        b_mean = b / np.maximum(1, counts)

        # compute the square root of the counts but clip so it's at least one
        bons = self.const / np.maximum(1, np.sqrt(counts))
        bounds = b_mean + bons

        i = np.argmax(bounds, axis=-1)
        j = np.argmin(counts, axis=-1)
        mask = (counts[np.arange(200), j] == 0)
        i[mask] = j[mask]

        a = np.zeros((self.batch_size, self.env.dim))
        a[np.arange(self.batch_size), i] = 1.0
        self.a = a
        return self.a


class BanditTransformerController(Controller):
    def __init__(self, model, sample=False,  batch_size=1):
        self.model = model
        self.du = model.config['action_dim']
        self.dx = model.config['state_dim']
        self.H = model.horizon
        self.sample = sample
        self.batch_size = batch_size
        
        self.name='Transformer'
        #self.zeros = torch.zeros(batch_size, self.dx**2 + self.du + 1).float().to(device)

    def set_env(self, env):
        return

    def set_batch_numpy_vec(self, batch):
        # Convert each element of the batch to a torch tensor
        new_batch = {}
        for key in batch.keys():
            new_batch[key] = torch.tensor(batch[key]).float().to(device)
        self.set_batch(new_batch)

    def act(self, x,return_probs=False):
        #self.batch['zeros'] = self.zeros

        states = torch.tensor(x)[None, :].float().to(device)
        self.batch['query_states'] = states

        a = self.model(self.batch)
        a = a.cpu().detach().numpy()[0]

        if self.sample:
            probs = scipy.special.softmax(a)
            i = np.random.choice(np.arange(self.du), p=probs)
        else:
            i = np.argmax(a)

        a = np.zeros(self.du)
        a[i] = 1.0
        
        if return_probs:
            return a, probs
        
        return a

    def act_numpy_vec(self, x):
        self.batch['zeros'] = self.zeros

        states = torch.tensor(np.array(x))
        if self.batch_size == 1:
            states = states[None,:]
        states = states.float().to(device)
        self.batch['query_states'] = states

        a = self.model(self.batch)
        a = a.cpu().detach().numpy()
        if self.batch_size == 1:
            a = a[0]

        if self.sample:
            probs = scipy.special.softmax(a, axis=-1)
            action_indices = np.array([np.random.choice(np.arange(self.du), p=p) for p in probs])
        else:
            action_indices = np.argmax(a, axis=-1)

        actions = np.zeros((self.batch_size, self.du))
        actions[np.arange(self.batch_size), action_indices] = 1.0
        return actions


class LinUCB(Controller):
    def __init__(self, env,  const=1.0):
        super().__init__()
        self.rand = True
        self.env=env
        self.t = 0
       
        self.theta = np.zeros(self.env.dim)
        self.cov = 1.0 * np.eye(self.env.dim)
        self.const = 2*const#const*(env.dim*np.log(env.H_context))**0.5
        
        self.name='LinUCB'
        
        
        
        
        
    
    def compare_actions_bernoulli(self,successes,failures):
        
        value_vec = (successes)/(1+failures+successes)
        
        bonus= 1/(1+failures+successes)**0.5
        
        value_vec=value_vec+bonus
        
        
        
        max_value = np.max(value_vec)
        
        index_list = np.where(value_vec==max_value)[0]
        
        index = np.random.choice(index_list)
        
        return index
        
        
        
        

    def act(self, x):
        
        
        
        
        if self.env.type=='bernoulli':
            hot_vector = np.zeros(self.env.act_num)
            
            if len(self.batch['context_rewards'][0]) < 1:
            
                index =self.compare_actions_bernoulli(np.zeros(self.env.act_num),np.zeros(self.env.act_num))
                hot_vector[index] = 1
                self.t += 1
            else:
                actions = self.batch['context_actions'].cpu().detach().numpy()[0]
                rewards = self.batch['context_rewards'].cpu().detach().numpy().flatten()

                actions_index=np.argmax(actions,axis=1)


                action_values=self.env.action_set[actions_index,:]
                
                successes = np.sum(action_values*rewards.reshape(-1,1),axis=0)
                failures = np.sum(action_values,axis=0)-successes
                index = self.compare_actions_bernoulli(successes,failures)
                
                hot_vector[index] = 1.
            return hot_vector
        
        
        
        
        
        
        
        
        
        
        
        if len(self.batch['context_rewards'][0]) < 1:
            hot_vector = np.zeros(self.env.act_num)
            index = np.random.choice(
                self.env.act_num, size=1, replace=False)
            hot_vector[index] = 1
            self.t += 1
            return hot_vector
        else:
            actions = self.batch['context_actions'].cpu().detach().numpy()[0]
            rewards = self.batch['context_rewards'].cpu().detach().numpy().flatten()
            
            actions_index=np.argmax(actions,axis=1)
          
            
            action_values=self.env.action_set[actions_index,:]

            cov = self.cov + action_values.T @ action_values
            cov_inv = np.linalg.inv(cov)

            theta = cov_inv @ action_values.T @ rewards.reshape(-1,1)

            if self.const == 0:
                index = np.argmax((self.env.action_set@theta).reshape(-1))
                self.a = np.zeros(self.env.act_num)
                self.a[index] = 1.0
                return self.a
            else:
                best_arm = None
                best_value = -np.inf
                for i in range(self.env.act_num):
                    value = np.sum(theta.reshape(-1)*self.env.action_set[i,:]) + \
                        self.const * np.sqrt(self.env.action_set[i:(i+1),:] @ cov_inv @ self.env.action_set[i:(i+1),:].T)
                    if value > best_value:
                        best_value = value
                        best_arm = i

                self.a = np.zeros(self.env.act_num)
                self.a[best_arm]=1.0
                return self.a


# def generate_k_hot_vectors(n, k):
#     indices = range(n)
#     k_hot_vectors = []
#     for combination in itertools.combinations(indices, k):
#         k_hot_vector = [0] * n
#         for index in combination:
#             k_hot_vector[index] = 1
#         k_hot_vectors.append(k_hot_vector)
#     return np.array(k_hot_vectors)
