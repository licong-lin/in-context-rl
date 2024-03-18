import argparse
import os
import pickle
import random
import torch
from tqdm import tqdm
import pdb

import gym
import numpy as np


import common_args
from envs import bandit_env


from ctrls.ctrl_bandit import (
    BanditTransformerController,
    GreedyOptPolicy,
    EmpMeanPolicy,
    OptPolicy,
    PessMeanPolicy,
    ThompsonSamplingPolicy,
    UCBPolicy,
    LinUCB,
    OffPolicy,
)
from utils import (
    build_bandit_data_filename,
    convert_to_tensor,
)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def rollin_bandit(env,  controller, orig=False):
    horizon = env.H_context
    opt_a_index = env.opt_a_index
    

    context_states = torch.zeros((1, horizon, env.dx)).float().to(device)
    context_actions = torch.zeros((1, horizon, env.du)).float().to(device)
    context_next_states = torch.zeros((1, horizon, env.dx)).float().to(device)
    context_rewards = torch.zeros((1, horizon, 1)).float().to(device)

    cum_means = []
    
        
    
    if controller in ['Transformer','UCB','LCB']:
        raise NotImplementedError
    elif controller=='Greedy':        ##GreedyOptPolicy
        controller = GreedyOptPolicy(
        env)
    elif controller=='Opt':            ##OptPolicy
        controller = OptPolicy(
        env)
    elif controller=='Emp':         ##EmpMeanPolicy
        controller = EmpMeanPolicy(
        env,online=True)
    elif controller=='LinUCB':   ##LinUCB
        controller = LinUCB(
        env)
    elif controller=='Thompson':   ##Thompson sampling
        controller = ThompsonSamplingPolicy(
                env,std=env.var)    ##env.var is std
        
    elif controller=='Off':   ##OffPolicy
        controller = OffPolicy(
        env,prob=np.ones(env.act_num)/env.act_num)    ##Uniform distribution
       
        
        
   

    
    
    
    
    
    for h in range(horizon):
        batch = {
            'context_states': context_states[:, :h, :],
            'context_actions': context_actions[:, :h, :],
            'context_next_states': context_next_states[:, :h, :],
            'context_rewards': context_rewards[:, :h, :],
        }

        controller.set_batch(batch)
        states_lnr, actions_lnr, next_states_lnr, rewards_lnr = env.deploy(
            controller)

        context_states[0, h, :] = convert_to_tensor(states_lnr[0])
        context_actions[0, h, :] = convert_to_tensor(actions_lnr[0])
        context_next_states[0, h, :] = convert_to_tensor(next_states_lnr[0])
        context_rewards[0, h, :] = convert_to_tensor(rewards_lnr[0])

        actions = actions_lnr.flatten()
        mean = env.get_arm_value(actions)

        cum_means.append(mean)
        
    context_states=context_states.cpu().detach().numpy().reshape(horizon,-1)
    context_actions=context_actions.cpu().detach().numpy().reshape(horizon,-1)
    context_next_states=context_next_states.cpu().detach().numpy().reshape(horizon,-1)
    context_rewards=context_rewards.cpu().detach().numpy().reshape(horizon,-1)





    
   
    return context_states,context_actions,context_next_states,context_rewards





# def rollin_bandit(env, cov, orig=False):
#     H = env.H_context
#     opt_a_index = env.opt_a_index
#     xs, us, xps, rs = [], [], [], []

#     exp = False
#     if exp == False:
#         cov = np.random.choice([0.0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0])
#         alpha = np.ones(env.dim)
#         probs = np.random.dirichlet(alpha)
#         probs2 = np.zeros(env.dim)
#         rand_index = np.random.choice(np.arange(env.dim))
#         probs2[rand_index] = 1.0
#         probs = (1 - cov) * probs + cov * probs2
#     else:
#         raise NotImplementedError

#     for h in range(H):
#         x = np.array([1])
#         u = np.zeros(env.dim)
#         i = np.random.choice(np.arange(env.dim), p=probs)
#         u[i] = 1.0
#         xp, r = env.transit(x, u)

#         xs.append(x)
#         us.append(u)
#         xps.append(xp)
#         rs.append(r)

#     xs, us, xps, rs = np.array(xs), np.array(us), np.array(xps), np.array(rs)
#     return xs, us, xps, rs







def generate_bandit_single_traj(env, controller, n_hists, n_samples):
    for j in range(n_hists):
        (
            context_states,
            context_actions,
            context_next_states,
            context_rewards,
        ) = rollin_bandit(env, controller)
        for k in range(n_samples):
            query_state = np.array([1])
            optimal_action = env.opt_a

            traj = {
                'query_state': query_state,
                'optimal_action': optimal_action,
                'context_states': context_states,
                'context_actions': context_actions,
                'context_next_states': context_next_states,
                'context_rewards': context_rewards,
                'means': env.means,
                'action_set': env.action_set
            }
    return traj

def generate_bandit_histories_from_envs(
        envs, controller, n_hists, n_samples, n_envs=None,
        dim=None, act_num=None, horizon=None, var=None, type=None,
        **kwargs
):
    trajs = []
    # added progress bar
    if envs is not None:
        raise
        for env in tqdm(envs):
            traj = generate_bandit_single_traj(env, controller, n_hists, n_samples)
            trajs.append(traj)
    elif n_envs is not None:
        for _ in tqdm(range(n_envs)):
        
            env = bandit_env.sample(dim, act_num, horizon, var, type=type)
            
            
            
             
            if var==0.05 and type=='bernoulli':   ##for mix A_0
                #pdb.set_trace()
                if _%2==0:
                    
                    traj = generate_bandit_single_traj(env, 'Off', n_hists, n_samples)
                   
                elif _%2==1:
                    
                    traj = generate_bandit_single_traj(env, 'Thompson', n_hists, n_samples)
                else:
                    raise 
                    
                trajs.append(traj)
                continue
            elif var==0.01 and type=='bernoulli':   ##for mix A_0
                #pdb.set_trace()
                if _%10!=0:
                    
                    traj = generate_bandit_single_traj(env, 'Off', n_hists, n_samples)
                   
                elif _%10==0:
                    
                    traj = generate_bandit_single_traj(env, 'Thompson', n_hists, n_samples)
                else:
                    raise 
                    
                trajs.append(traj)
                continue
                
                
                
            
            pdb.set_trace()
            traj = generate_bandit_single_traj(env, controller, n_hists, n_samples)
            trajs.append(traj)
            
    return trajs





def generate_bandit_histories(n_envs, dim, act_num,horizon, var, type, **kwargs):
    # envs = [bandit_env.sample(dim, act_num,horizon, var, type=type)
    #         for _ in range(n_envs)]

    trajs = generate_bandit_histories_from_envs(
        envs=None,
        **kwargs,
        n_envs=n_envs, dim=dim, act_num=act_num, horizon=horizon, var=var, type=type,
    )
    return trajs





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    common_args.add_dataset_args(parser)

    args = vars(parser.parse_args())
    print("Args: ", args)

    env = args['env']
    n_envs = args['envs']
    n_eval_envs = args['envs_eval']
    n_hists = args['hists']
    n_samples = args['samples']
    horizon = args['H']
    dim = args['dim']
    act_num = args['act_num']
    var = args['var']
    controller=args['controller']
    

    
    #cov = args['cov']

    n_train_envs = int(.8 * n_envs)
    n_test_envs = n_envs - n_train_envs

    config = {
        'n_hists': n_hists,
        'n_samples': n_samples,
        'horizon': horizon,
    }

    
    
    
    
    
    
    rep_num=10 
    
    
    
    for iii in range(rep_num):
    
        if env == 'bandit':
            config.update({'dim': dim, 'act_num':act_num, 'var': var, 'controller': controller, 'type': 'uniform'})

            train_trajs = generate_bandit_histories(n_train_envs//rep_num, **config)
            test_trajs = generate_bandit_histories(n_test_envs//rep_num, **config)
            eval_trajs = generate_bandit_histories(n_eval_envs//rep_num, **config)

            train_filepath = build_bandit_data_filename(env, n_envs, config, mode=0)
            test_filepath = build_bandit_data_filename(env, n_envs, config, mode=1)
            eval_filepath = build_bandit_data_filename(env, n_eval_envs, config, mode=2)

        elif env == 'bandit_bernoulli':

            #raise NotImplementedError
            config.update({'dim': dim,  'var': var, 'act_num':act_num,'controller': controller, 'type': 'bernoulli'})

            train_trajs = generate_bandit_histories(n_train_envs//rep_num, **config)
            test_trajs = generate_bandit_histories(n_test_envs//rep_num, **config)
            eval_trajs = generate_bandit_histories(n_eval_envs//rep_num, **config)

            train_filepath = build_bandit_data_filename(env, n_envs, config, mode=0)
            test_filepath = build_bandit_data_filename(env, n_envs, config, mode=1)
            eval_filepath = build_bandit_data_filename(env, n_eval_envs, config, mode=2)



        else:
            raise NotImplementedError




        if not os.path.exists('/username/datasets'):                ## save result to username
            os.makedirs('/username/datasets', exist_ok=True)
        with open(train_filepath[:-4]+'_'+str(iii)+'_.pkl', 'wb') as file:
            pickle.dump(train_trajs, file)
        with open(test_filepath[:-4]+'_'+str(iii)+'_.pkl', 'wb') as file:
            pickle.dump(test_trajs, file)
        with open(eval_filepath[:-4]+'_'+str(iii)+'_.pkl', 'wb') as file:
            pickle.dump(eval_trajs, file)    

        
        

    # if not os.path.exists('datasets'):
    #     os.makedirs('datasets', exist_ok=True)
    # with open(train_filepath, 'wb') as file:
    #     pickle.dump(train_trajs, file)
    # with open(test_filepath, 'wb') as file:
    #     pickle.dump(test_trajs, file)
    # with open(eval_filepath, 'wb') as file:
    #     pickle.dump(eval_trajs, file)

    print(f"Saved to {train_filepath}.")
    print(f"Saved to {test_filepath}.")
    print(f"Saved to {eval_filepath}.")
