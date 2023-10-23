import argparse
import os

import matplotlib.pyplot as plt
import torch
from torchvision.transforms import transforms

import common_args
from dataset import Dataset, ImageDataset
from net import Transformer, ImageTransformer
from utils import (
    build_bandit_data_filename,
    build_bandit_model_filename,
    build_darkroom_data_filename,
    build_darkroom_model_filename,
    build_miniworld_data_filename,
    build_miniworld_model_filename,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
    if not os.path.exists('figs/loss'):
        os.makedirs('figs/loss', exist_ok=True)
    if not os.path.exists('models'):
        os.makedirs('models', exist_ok=True)

    parser = argparse.ArgumentParser()
    common_args.add_dataset_args(parser)
    common_args.add_model_args(parser)
    common_args.add_train_args(parser)

    args = vars(parser.parse_args())
    print("Args: ", args)

    env = args['env']
    n_envs = args['envs']
    n_hists = args['hists']
    n_samples = args['samples']
    horizon = args['H']
    dim = args['dim']
   
    n_embd = args['embd']
    n_head = args['head']
    n_layer = args['layer']
    lr = args['lr']
    shuffle = args['shuffle']
    dropout = args['dropout']
    var = args['var']
    act_num = args['act_num']
    state_dim = dim
    action_dim = act_num
    controller = args['controller']
    imit = args['imit']
    act_type = args['act_type']


    
    num_epochs = args['num_epochs']

    dataset_config = {
        'n_hists': n_hists,
        'n_samples': n_samples,
        'horizon': horizon,
        'dim': dim,
    }
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
        'act_type':act_type,
    }
    if env == 'bandit':
        state_dim = 1

        dataset_config.update({'var': var, 'act_num': act_num, 'controller': controller, 'type': 'uniform'})
        path_train = build_bandit_data_filename(
            env, n_envs, dataset_config, mode=0)
        path_test = build_bandit_data_filename(
            env, n_envs, dataset_config, mode=1)

        model_config.update({'var': var})
        filename = build_bandit_model_filename(env, model_config)

    elif env == 'bandit_bernoulli':
        state_dim = 1

        dataset_config.update({'var': var, 'act_num': act_num, 'controller': controller, 'type': 'bernoulli'})
        path_train = build_bandit_data_filename(
            env, n_envs, dataset_config, mode=0)
        path_test = build_bandit_data_filename(
            env, n_envs, dataset_config, mode=1)

        model_config.update({'var': var})
        filename = build_bandit_model_filename(env, model_config)
        
     
    elif env.startswith('darkroom'):
        state_dim = 2
        action_dim = 5

        dataset_config.update({'rollin_type': 'uniform'})
        path_train = build_darkroom_data_filename(
            env, n_envs, dataset_config, mode=0)
        path_test = build_darkroom_data_filename(
            env, n_envs, dataset_config, mode=1)

        filename = build_darkroom_model_filename(env, model_config)

    elif env == 'miniworld':
        state_dim = 2   # direction vector is 2D
        action_dim = 4

        dataset_config.update({'rollin_type': 'uniform'})
        path_train = build_miniworld_data_filename(
            env, n_envs, dataset_config, mode=0)
        path_test = build_miniworld_data_filename(
            env, n_envs, dataset_config, mode=1)

        filename = build_miniworld_model_filename(env, model_config)

    else:
        raise NotImplementedError

    config = {
        'horizon': horizon,
        'dim':dim,
        'act_num':act_num,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'n_layer': n_layer,
        'n_embd': n_embd,
        'n_head': n_head,
        'shuffle': shuffle,
        'dropout': dropout,
        'test': False,
        'act_type': act_type,

    }
    if env == 'miniworld':
        config.update({'image_size': 25})
        model = ImageTransformer(config).to(device)
    else:
        model = Transformer(config).to(device)

    params = {
        'batch_size': 128,
        'shuffle': True,
    }

    if env == 'miniworld':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        train_dataset = ImageDataset(path_train, config, transform)
        test_dataset = ImageDataset(path_test, config, transform)
    else:
        train_dataset = Dataset(path_train, config)
        test_dataset = Dataset(path_test, config)
    train_loader = torch.utils.data.DataLoader(train_dataset, **params)
    test_loader = torch.utils.data.DataLoader(test_dataset, **params)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    test_loss = []
    train_loss = []
    
    
    
    
    
#     model = Transformer(config).to(device)
#     model_path = f'models/{filename}_epoch{1150}.pt'
#     opt_path = f'models/{filename}_epoch{1150}_opt.pt'


#     checkpoint = torch.load(model_path)
#     model.load_state_dict(checkpoint)
#     checkpoint_2 = torch.load(opt_path)
#     optimizer.load_state_dict(checkpoint_2)
#     now_epoch=1150
    
#     for epoch in range(now_epoch+1,num_epochs):
   
    

    for epoch in range(num_epochs):
        # EVALUATION
        with torch.no_grad():
            epoch_test_loss = 0.0
            for i, batch in enumerate(test_loader):
                true_actions = batch['optimal_actions']
                imit_actions=batch['context_actions']


                pred_actions = model(batch)
                true_actions = true_actions.unsqueeze(
                    1).repeat(1, pred_actions.shape[1], 1)
              


                true_actions = true_actions.reshape(-1, action_dim)
                pred_actions = pred_actions.reshape(-1, action_dim)
                imit_actions = imit_actions.reshape(-1,action_dim)     
               

                if imit=='true':
                    loss = loss_fn(pred_actions, true_actions)
                else:
                    loss = loss_fn(pred_actions, imit_actions)
                    
                epoch_test_loss += loss.item() / horizon

        test_loss.append(epoch_test_loss / len(test_dataset))

        # TRAINING
        epoch_train_loss = 0.0
        for i, batch in enumerate(train_loader):
            true_actions = batch['optimal_actions']
            imit_actions=batch['context_actions']
            pred_actions = model(batch)
            true_actions = true_actions.unsqueeze(
                1).repeat(1, pred_actions.shape[1], 1)
            
            true_actions = true_actions.reshape(-1, action_dim)
            pred_actions = pred_actions.reshape(-1, action_dim)
            imit_actions = imit_actions.reshape(-1,action_dim)



            optimizer.zero_grad()
            if imit=='true':
                loss = loss_fn(pred_actions, true_actions)
            else:
                loss = loss_fn(pred_actions, imit_actions)
                
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() / horizon

        train_loss.append(epoch_train_loss / len(train_dataset))

        # LOGGING
        
        if epoch==0:
            torch.save(model.state_dict(),
                       f'models/{filename}_epoch{epoch+1}.pt')
            torch.save(optimizer.state_dict(),
                       f'models/{filename}_epoch{epoch+1}_opt.pt')
            print('save first one')
            
            
        if n_envs<=100000:   
            
            if (epoch + 1) % 50 == 0:
                torch.save(model.state_dict(),
                           f'models/{filename}_epoch{epoch+1}.pt')

                torch.save(optimizer.state_dict(),
                           f'models/{filename}_epoch{epoch+1}_opt.pt')
        else:
            if (epoch + 1) % 10 == 0:
                torch.save(model.state_dict(),
                           f'models/{filename}_epoch{epoch+1}.pt')

                torch.save(optimizer.state_dict(),
                           f'models/{filename}_epoch{epoch+1}_opt.pt')
            
            

        # PLOTTING
        if (epoch + 1) % 2 == 0:
            print(f"Epoch: {epoch + 1}")
            print(f"Test Loss:        {test_loss[-1]}")
            print(f"Train Loss:       {train_loss[-1]}")
            print("\n")

            plt.yscale('log')
            plt.plot(train_loss[1:], label="Train Loss")
            plt.plot(test_loss[1:], label="Test Loss")
            plt.legend()
            plt.savefig(f"figs/loss/{filename}_train_loss.png")
            plt.clf()

    torch.save(model.state_dict(), f'models/{filename}.pt')
    print("Done.")
