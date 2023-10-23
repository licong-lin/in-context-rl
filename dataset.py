import pickle

import numpy as np
import torch
import os

from utils import convert_to_tensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Dataset(torch.utils.data.Dataset):
    """Dataset class."""

    def __init__(self, path, config):
        self.shuffle = config['shuffle']
        self.horizon = config['horizon']
        
        
        if os.path.exists(path):
            
                
            with open(path, 'rb') as f:
                combined_list = []
                ii=0
                while True:

                    try:
                        ii+=1


                        loaded_list = pickle.load(f)

                        combined_list.extend(loaded_list)
                        
                    except EOFError:
                       
                        break  


                self.trajs = combined_list
                
        else:
            combined_list = []
            for iii in range(10):
                path_current=path[:-4]+'_'+str(iii)+'_.pkl'
                with open(path_current, 'rb') as f:
                    
                    loaded_list = pickle.load(f)

                    combined_list.extend(loaded_list)
            
            self.trajs = combined_list
                 
                


        

        # with open(path, 'rb') as f:             ##previous
        #     self.trajs = pickle.load(f)

        
        
        action_set= []
        
        context_states = []
        context_actions = []
        context_next_states = []
        context_rewards = []
        query_states = []
        optimal_actions = []

        for traj in self.trajs:
            context_states.append(traj['context_states'])
            context_actions.append(traj['context_actions'])
            context_next_states.append(traj['context_next_states'])
            context_rewards.append(traj['context_rewards'])

            query_states.append(traj['query_state'])
            optimal_actions.append(traj['optimal_action'])
            
            action_set.append(traj['action_set'])

        context_states = np.array(context_states)
        context_actions = np.array(context_actions)
        context_next_states = np.array(context_next_states)
        context_rewards = np.array(context_rewards)
        if len(context_rewards.shape) < 3:
            context_rewards = context_rewards[:, :, None]
        query_states = np.array(query_states)
        optimal_actions = np.array(optimal_actions)
        
        action_set=np.array(action_set)

        self.dataset = {
            'query_states': convert_to_tensor(query_states),
            'optimal_actions': convert_to_tensor(optimal_actions),
            'context_states': convert_to_tensor(context_states),
            'context_actions': convert_to_tensor(context_actions),
            'context_next_states': convert_to_tensor(context_next_states),
            'context_rewards': convert_to_tensor(context_rewards),
            'action_set': convert_to_tensor(action_set)
        }

        self.zeros = torch.zeros(
            config['state_dim'] ** 2 + config['action_dim'] + 1
        ).float().to(device)

    def __len__(self):
        'Denotes the total number of samples'
        
        assert len(self.dataset['context_states'])==len(self.dataset['query_states'])
        return len(self.dataset['context_states'])

    def __getitem__(self, index):
        'Generates one sample of data'
        res = {
            'context_states': self.dataset['context_states'][index],
            'context_actions': self.dataset['context_actions'][index],
            'context_next_states': self.dataset['context_next_states'][index],
            'context_rewards': self.dataset['context_rewards'][index],
            'query_states': self.dataset['query_states'][index],
            'optimal_actions': self.dataset['optimal_actions'][index],
            
            'action_set': self.dataset['action_set'][index],
            'zeros': self.zeros,
        }
        
        '''
        if self.shuffle:   ## no shuffle when using online collected pretrainign data
            perm = torch.randperm(self.horizon)
            res['context_states'] = res['context_states'][perm]
            res['context_actions'] = res['context_actions'][perm]
            res['context_next_states'] = res['context_next_states'][perm]
            res['context_rewards'] = res['context_rewards'][perm]
        '''

        return res


class ImageDataset(Dataset):
    """"Dataset class for image-based data."""

    def __init__(self, path, config, transform):
        super().__init__(path, config)
        self.transform = transform

        context_filepaths = []
        query_images = []

        for traj in self.trajs:
            context_filepaths.append(traj['context_images'])
            query_image = self.transform(traj['query_image']).float()
            query_images.append(query_image)

        self.dataset.update({
            'context_filepaths': context_filepaths,
            'query_images': torch.stack(query_images),
        })

    def __getitem__(self, index):
        'Generates one sample of data'
        filepath = self.dataset['context_filepaths'][index]
        context_images = np.load(filepath)
        context_images = [self.transform(images) for images in context_images]
        context_images = torch.stack(context_images).float()

        query_images = self.dataset['query_images'][index]

        res = {
            'context_images': context_images.to(device),
            'context_states': self.dataset['context_states'][index],
            'context_actions': self.dataset['context_actions'][index],
            'context_next_states': self.dataset['context_next_states'][index],
            'context_rewards': self.dataset['context_rewards'][index],
            'query_images': query_images.to(device),
            'query_states': self.dataset['query_states'][index],
            'optimal_actions': self.dataset['optimal_actions'][index],
            'zeros': self.zeros,
        }

        if self.shuffle:
            perm = torch.randperm(self.horizon)
            res['context_images'] = res['context_images'][perm]
            res['context_states'] = res['context_states'][perm]
            res['context_actions'] = res['context_actions'][perm]
            res['context_next_states'] = res['context_next_states'][perm]
            res['context_rewards'] = res['context_rewards'][perm]

        return res
