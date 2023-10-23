import argparse
import os
import pickle
import random
import torch
from tqdm import tqdm
import pdb

import gym
import numpy as np
from skimage.transform import resize

import common_args
from envs import darkroom_env, bandit_env


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
    build_darkroom_data_filename,
    build_miniworld_data_filename,
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


def rollin_mdp(env, rollin_type):
    states = []
    actions = []
    next_states = []
    rewards = []

    state = env.reset()
    for _ in range(env.horizon):
        if rollin_type == 'uniform':
            state = env.sample_state()
            action = env.sample_action()
        elif rollin_type == 'expert':
            action = env.opt_action(state)
        else:
            raise NotImplementedError
        next_state, reward = env.transit(state, action)

        states.append(state)
        actions.append(action)
        next_states.append(next_state)
        rewards.append(reward)
        state = next_state

    states = np.array(states)
    actions = np.array(actions)
    next_states = np.array(next_states)
    rewards = np.array(rewards)

    return states, actions, next_states, rewards


def rand_pos_and_dir(env):
    pos_vec = np.random.uniform(0, env.size, size=3)
    pos_vec[1] = 0.0
    dir_vec = np.random.uniform(0, 2 * np.pi)
    return pos_vec, dir_vec


def rollin_mdp_miniworld(env, horizon, rollin_type, target_shape=(25, 25, 3)):
    observations = []
    pos_and_dirs = []
    actions = []
    rewards = []

    for _ in range(horizon):
        if rollin_type == 'uniform':
            init_pos, init_dir = rand_pos_and_dir(env)
            env.place_agent(pos=init_pos, dir=init_dir)

        obs = env.render_obs()
        obs = resize(obs, target_shape, anti_aliasing=True)
        observations.append(obs)
        pos_and_dirs.append(np.concatenate(
            [env.agent.pos[[0, -1]], env.agent.dir_vec[[0, -1]]]))

        if rollin_type == 'uniform':
            action = np.random.randint(env.action_space.n)
        elif rollin_type == 'expert':
            action = env.opt_a(obs, env.agent.pos, env.agent.dir_vec)
        else:
            raise ValueError("Invalid rollin type")
        _, rew, _, _, _ = env.step(action)
        a_zero = np.zeros(env.action_space.n)
        a_zero[action] = 1

        actions.append(a_zero)
        rewards.append(rew)

    observations = np.array(observations)
    states = np.array(pos_and_dirs)[..., 2:]    # only use dir
    actions = np.array(actions)
    rewards = np.array(rewards)
    return observations, states, actions, rewards


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


def generate_mdp_histories_from_envs(envs, n_hists, n_samples, rollin_type):
    trajs = []
    for env in envs:
        for j in range(n_hists):
            (
                context_states,
                context_actions,
                context_next_states,
                context_rewards,
            ) = rollin_mdp(env, rollin_type=rollin_type)
            for k in range(n_samples):
                query_state = env.sample_state()
                optimal_action = env.opt_action(query_state)

                traj = {
                    'query_state': query_state,
                    'optimal_action': optimal_action,
                    'context_states': context_states,
                    'context_actions': context_actions,
                    'context_next_states': context_next_states,
                    'context_rewards': context_rewards,
                    'goal': env.goal,
                }

                # Add perm_index for DarkroomEnvPermuted
                if hasattr(env, 'perm_index'):
                    traj['perm_index'] = env.perm_index

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


def generate_darkroom_histories(goals, dim, horizon, **kwargs):
    envs = [darkroom_env.DarkroomEnv(dim, goal, horizon) for goal in goals]
    trajs = generate_mdp_histories_from_envs(envs, **kwargs)
    return trajs


def generate_darkroom_permuted_histories(indices, dim, horizon, **kwargs):
    envs = [darkroom_env.DarkroomEnvPermuted(
        dim, index, horizon) for index in indices]
    trajs = generate_mdp_histories_from_envs(envs, **kwargs)
    return trajs


def generate_miniworld_histories(env_ids, image_dir, n_hists, n_samples, horizon, target_shape, rollin_type='uniform'):
    if not os.path.exists(image_dir):
        os.makedirs(image_dir, exist_ok=True)

    n_envs = len(env_ids)
    env = gym.make('MiniWorld-OneRoomS6FastMultiFourBoxesFixedInit-v0')
    obs = env.reset()

    trajs = []
    for i, env_id in enumerate(env_ids):
        print(f"Generating histories for env {i}/{n_envs}")
        env.set_task(env_id)
        env.reset()
        for j in range(n_hists):
            (
                context_images,
                context_states,
                context_actions,
                context_rewards,
            ) = rollin_mdp_miniworld(env, horizon, rollin_type=rollin_type, target_shape=target_shape)
            filepath = f'{image_dir}/context{i}_{j}.npy'
            np.save(filepath, context_images)

            for _ in range(n_samples):
                init_pos, init_dir = rand_pos_and_dir(env)
                env.place_agent(pos=init_pos, dir=init_dir)
                obs = env.render_obs()
                obs = resize(obs, target_shape, anti_aliasing=True)

                action = env.opt_a(obs, env.agent.pos, env.agent.dir_vec)
                one_hot_action = np.zeros(env.action_space.n)
                one_hot_action[action] = 1

                traj = {
                    'query_image': obs,
                    'query_state': env.agent.dir_vec[[0, -1]],
                    'optimal_action': one_hot_action,
                    'context_images': filepath,
                    'context_states': context_states,
                    'context_actions': context_actions,
                    'context_next_states': context_states,  # unused
                    'context_rewards': context_rewards,
                }
                trajs.append(traj)
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




        elif env == 'darkroom_heldout':
            config.update({'dim': dim, 'rollin_type': 'uniform'})
            goals = np.array([[(j, i) for i in range(dim)]
                             for j in range(dim)]).reshape(-1, 2)
            np.random.RandomState(seed=0).shuffle(goals)
            train_test_split = int(.8 * len(goals))
            train_goals = goals[:train_test_split]
            test_goals = goals[train_test_split:]

            eval_goals = np.array(test_goals.tolist() *
                                  int(100 // len(test_goals)))
            train_goals = np.repeat(train_goals, n_envs // (dim * dim), axis=0)
            test_goals = np.repeat(test_goals, n_envs // (dim * dim), axis=0)

            train_trajs = generate_darkroom_histories(train_goals, **config)
            test_trajs = generate_darkroom_histories(test_goals, **config)
            eval_trajs = generate_darkroom_histories(eval_goals, **config)

            train_filepath = build_darkroom_data_filename(
                env, n_envs, config, mode=0)
            test_filepath = build_darkroom_data_filename(
                env, n_envs, config, mode=1)
            eval_filepath = build_darkroom_data_filename(env, 100, config, mode=2)

        elif env == 'darkroom_permuted':
            config.update({'dim': dim, 'rollin_type': 'uniform'})
            indices = np.arange(120)    # 5! permutations in darkroom
            np.random.RandomState(seed=0).shuffle(indices)
            train_test_split = int(.8 * len(indices))
            train_indices = indices[:train_test_split]
            test_indices = indices[train_test_split:]

            eval_indices = np.array(test_indices.tolist() *
                                    int(100 // len(test_indices)))
            train_indices = np.repeat(
                train_indices, n_train_envs // len(train_indices), axis=0)
            test_indices = np.repeat(
                test_indices, n_test_envs // len(test_indices), axis=0)

            train_trajs = generate_darkroom_permuted_histories(
                train_indices, **config)
            test_trajs = generate_darkroom_permuted_histories(
                test_indices, **config)
            eval_trajs = generate_darkroom_permuted_histories(
                eval_indices, **config)

            train_filepath = build_darkroom_data_filename(
                env, n_envs, config, mode=0)
            test_filepath = build_darkroom_data_filename(
                env, n_envs, config, mode=1)
            eval_filepath = build_darkroom_data_filename(env, 100, config, mode=2)

        elif env == 'miniworld':
            import gymnasium as gym
            import miniworld

            config.update({'rollin_type': 'uniform', 'target_shape': (25, 25, 3)})

            env_ids = np.arange(n_envs)
            train_test_split = int(.8 * len(env_ids))
            train_env_ids = env_ids[:train_test_split]
            test_env_ids = env_ids[train_test_split:]
            train_env_ids = np.repeat(train_env_ids, n_envs // len(env_ids), axis=0)
            test_env_ids = np.repeat(test_env_ids, n_envs // len(env_ids), axis=0)

            train_filepath = build_miniworld_data_filename(
                env, n_envs, config, mode=0)
            test_filepath = build_miniworld_data_filename(
                env, n_envs, config, mode=1)
            eval_filepath = build_miniworld_data_filename(env, 100, config, mode=2)

            train_trajs = generate_miniworld_histories(
                train_env_ids,
                train_filepath.split('.')[0],
                **config)
            test_trajs = generate_miniworld_histories(
                test_env_ids,
                test_filepath.split('.')[0],
                **config)
            eval_trajs = generate_miniworld_histories(
                test_env_ids[:100],
                eval_filepath.split('.')[0],
                **config)

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
