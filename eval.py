import argparse
import os
import pickle

import matplotlib.pyplot as plt
import torch
import pdb

import common_args
from evals import eval_bandit, eval_darkroom
from net import Transformer, ImageTransformer
from utils import (
    build_bandit_data_filename,
    build_bandit_model_filename,
    build_darkroom_data_filename,
    build_darkroom_model_filename,
    build_miniworld_data_filename,
    build_miniworld_model_filename,
)
import numpy as np
import scipy
from envs import darkroom_env, bandit_env

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    common_args.add_dataset_args(parser)
    common_args.add_model_args(parser)
    common_args.add_eval_args(parser)

    args = vars(parser.parse_args())
    print("Args: ", args)

    n_envs = args['envs']
    n_hists = args['hists']
    n_samples = args['samples']
    H = args['H']
    dim = args['dim']
    
    act_num = args['act_num']
    controller = args['controller']
    state_dim = dim
    action_dim = act_num
    
    n_embd = args['embd']
    n_head = args['head']
    n_layer = args['layer']
    lr = args['lr']
    epoch = args['epoch']
    shuffle = args['shuffle']
    dropout = args['dropout']
    var = args['var']
   


    test_cov = args['test_cov']
    envname = args['env']
    horizon = args['hor']
    n_eval = args['n_eval']
    
    imit=args['imit']
    act_type=args['act_type']
    #act_type='relu'



  
    if horizon < 0:
        horizon = H    ##default 100

    model_config = {
        'shuffle': shuffle,
        'lr': lr,
        'dropout': dropout,
        'n_embd': n_embd,
        'n_layer': n_layer,
        'n_head': n_head,
        'n_envs': n_envs,
        'n_hists': n_hists,
        'n_samples': n_samples,
        'horizon': horizon,
        'dim': dim,
        'act_num':act_num,
        'controller': controller,
        'imit':imit,
        'act_type': act_type,


    }
    if envname == 'bandit':
        state_dim = 1

        model_config.update({'var': var})
        filename = build_bandit_model_filename(envname, model_config)
        bandit_type = 'uniform'
    elif envname == 'bandit_bernoulli':
       
        state_dim = 1

        model_config.update({'var': var})
        filename = build_bandit_model_filename(envname, model_config)
        bandit_type = 'bernoulli'
        
    elif envname.startswith('darkroom'):
        state_dim = 2
        action_dim = 5

        filename = build_darkroom_model_filename(envname, model_config)
    elif envname == 'miniworld':
        state_dim = 2
        action_dim = 4

        filename = build_miniworld_model_filename(envname, model_config)
    else:
        raise NotImplementedError

   
    
    
    if envname=='bandit':
        
        eval_trajs = [bandit_env.sample(dim, act_num,horizon, var, type='uniform')
                for _ in range(n_eval)]
    elif envname=='bandit_bernoulli':
        eval_trajs = [bandit_env.sample(dim, act_num,horizon, var, type='bernoulli')
                for _ in range(n_eval)]
    else:
        raise
    
    
    
    
    config = {
        'horizon': H,
        'act_num':act_num,
        'dim': dim,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'n_layer': n_layer,
        'n_embd': n_embd,
        'n_head': n_head,
        'dropout': dropout,
        'act_type':act_type,
        'test': True,
    }
    
    

    # Load network from saved file.
    # By default, load the final file, otherwise load specified epoch.
    if envname == 'miniworld':
        config.update({'image_size': 25})
        model = ImageTransformer(config).to(device)
    else:
        model = Transformer(config).to(device)
    if epoch < 0:
        model_path = f'models/{filename}.pt'
    else:
        model_path = f'models/{filename}_epoch{epoch}.pt'
    checkpoint = torch.load(model_path,map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    model.eval()

    dataset_config = {
        'horizon': horizon,
        'dim': dim,
    }
    if envname in ['bandit']:
        
        dataset_config.update({'var': var,  'act_num' : act_num, 'controller': controller,'type': 'uniform'})
        eval_filepath = build_bandit_data_filename(
            envname, n_eval, dataset_config, mode=2)
        
        save_filename = f'{filename}_hor{horizon}_test.pkl'
       
    elif envname in [ 'bandit_bernoulli']:
        dataset_config.update({'var': var,  'act_num' : act_num, 'controller': controller,'type': 'bernoulli'})
        eval_filepath = build_bandit_data_filename(
            envname, n_eval, dataset_config, mode=2)
        
        save_filename = f'{filename}_hor{horizon}_test.pkl'
        
        
        
        
    elif envname in ['darkroom_heldout', 'darkroom_permuted']:
        dataset_config.update({'rollin_type': 'uniform'})
        eval_filepath = build_darkroom_data_filename(
            envname, n_eval, dataset_config, mode=2)
        save_filename = f'{filename}_hor{horizon}.pkl'
    elif envname == 'miniworld':
        dataset_config.update({'rollin_type': 'uniform'})
        eval_filepath = build_miniworld_data_filename(
            envname, n_eval, dataset_config, mode=2)
        save_filename = f'{filename}_hor{horizon}.pkl'
    else:
        raise ValueError(f'Environment {envname} not supported')

    # Load evaluation trajectories
     
         
          
                    
#     if os.path.exists(eval_filepath):
    
#         with open(eval_filepath, 'rb') as f:
#             combined_list = []
#             while True:
#                 ii=0
#                 try:
#                     ii+=1



#                     loaded_list = pickle.load(f)


#                     combined_list.extend(loaded_list)
#                 except EOFError:
#                     print('load times:',ii)


#                     break

#             eval_trajs=combined_list
      
#     else:
           
#         combined_list = []
#         for iii in range(10):
#             path_current=eval_filepath[:-4]+'_'+str(iii)+'_.pkl'
#             with open(path_current, 'rb') as f:

#                 loaded_list = pickle.load(f)

#                 combined_list.extend(loaded_list)
                
        
#         eval_trajs=combined_list
        
#         print(len(eval_trajs))
                
                


    
#     with open(eval_filepath, 'rb') as f:   ##previous
#         eval_trajs = pickle.load(f)
        
        
        
        
        

    n_eval = min(n_eval, len(eval_trajs))

    evals_filename = f"evals_epoch{epoch}"
    if not os.path.exists(f'figs/{evals_filename}'):
        os.makedirs(f'figs/{evals_filename}', exist_ok=True)
    if not os.path.exists(f'figs/{evals_filename}/bar'):
        os.makedirs(f'figs/{evals_filename}/bar', exist_ok=True)
    if not os.path.exists(f'figs/{evals_filename}/online'):
        os.makedirs(f'figs/{evals_filename}/online', exist_ok=True)
    if not os.path.exists(f'figs/{evals_filename}/graph'):
        os.makedirs(f'figs/{evals_filename}/graph', exist_ok=True)

    # Online and offline evaluation.
    if envname == 'bandit' or envname == 'bandit_bernoulli':
        config = {
            'horizon': horizon,
            'var': var,
            'n_eval': n_eval,
            'bandit_type': bandit_type,
            'mytest': True
        }
        
        all_means,all_means_diff,means,sems=eval_bandit.online(eval_trajs, model, **config)
        
        my_save_path=f"my_plots/envname_{envname}_n_sam_{n_samples}_var_{var}_ctrl_{controller}_dim_{dim}_act_{act_num}_\
                            imit_{imit}_net_{[n_layer,n_head,act_type,n_embd]}_epoch_{epoch}_lr_{lr}.pkl"
        
        #my_save_path='my_plots/record_6.pkl'
        
        with open(my_save_path, 'wb') as f:
            pickle.dump([all_means,all_means_diff,means,sems ], f)
        
  
        
        

        
        
        
        
        
        
        

    elif envname in ['darkroom_heldout', 'darkroom_permuted']:
        config = {
            'Heps': 40,
            'horizon': horizon,
            'H': H,
            'n_eval': min(20, n_eval),
            'dim': dim,
            'permuted': True if envname == 'darkroom_permuted' else False,
        }
        eval_darkroom.online(eval_trajs, model, **config)
        plt.savefig(f'figs/{evals_filename}/online/{save_filename}.png')
        plt.clf()

        del config['Heps']
        del config['horizon']
        config['n_eval'] = n_eval
        eval_darkroom.offline(eval_trajs, model, **config)
        plt.savefig(f'figs/{evals_filename}/bar/{save_filename}_bar.png')
        plt.clf()

    elif envname == 'miniworld':
        from evals import eval_miniworld
        save_video = args['save_video']
        filename_prefix = f'videos/{save_filename}/{evals_filename}/'
        config = {
            'Heps': 40,
            'horizon': horizon,
            'H': H,
            'n_eval': min(20, n_eval),
            'save_video': save_video,
            'filename_template': filename_prefix + '{controller}_env{env_id}_ep{ep}_online.gif',
        }

        if save_video and not os.path.exists(f'videos/{save_filename}/{evals_filename}'):
            os.makedirs(
                f'videos/{save_filename}/{evals_filename}', exist_ok=True)

        eval_miniworld.online(eval_trajs, model, **config)
        plt.savefig(f'figs/{evals_filename}/online/{save_filename}.png')
        plt.clf()

        del config['Heps']
        del config['horizon']
        del config['H']
        config['n_eval'] = n_eval
        config['filename_template'] = filename_prefix + \
            '{controller}_env{env_id}_offline.gif'
        eval_miniworld.offline(eval_trajs, model, **config)
        plt.savefig(f'figs/{evals_filename}/bar/{save_filename}_bar.png')
        plt.clf()
