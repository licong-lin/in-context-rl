import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_bandit_data_filename(env, n_envs, config, mode):
    """
    Builds the filename for the bandit data.
    Mode is either 0: train, 1: test, 2: eval.
    """
    
    
    filename_template = 'datasets/trajs_{}.pkl' 


    
    #filename_template = 'username/datasets/trajs_{}.pkl'     ##enter the username
    
    
    
    
    filename = env
    filename += '_envs' + str(n_envs)
    if mode != 2:
        filename += '_hists' + str(config['n_hists'])
        filename += '_samples' + str(config['n_samples'])
    filename += '_H' + str(config['horizon'])
    filename += '_d' + str(config['dim'])
    filename += '_act_num' + str(config['act_num'])
    filename += '_var' + str(config['var'])
    filename += '_controller' + str(config['controller'])
    if mode == 0:
        filename += '_train'
    elif mode == 1:
        filename += '_test'
    elif mode == 2:
        filename += '_eval'
    return filename_template.format(filename)


def build_bandit_model_filename(env, config):
    """
    Builds the filename for the bandit model.
    """
    filename = env
    filename += '_shuf' + str(config['shuffle'])
    filename += '_lr' + str(config['lr'])
    filename += '_do' + str(config['dropout'])
    filename += '_embd' + str(config['n_embd'])
    filename += '_layer' + str(config['n_layer'])
    filename += '_head' + str(config['n_head'])
    filename += '_envs' + str(config['n_envs'])
    filename += '_hists' + str(config['n_hists'])
    filename += '_samples' + str(config['n_samples'])
    filename += '_var' + str(config['var'])
    filename += '_controller' + str(config['controller'])
    filename += '_H' + str(config['horizon'])
    filename += '_d' + str(config['dim'])
    filename += '_act_num' + str(config['act_num'])
    filename += 'imit' + str(config['imit'])
    filename += 'act_type' + str(config['act_type'])
    return filename







def convert_to_tensor(x):
    return torch.tensor(np.asarray(x)).float().to(device)
