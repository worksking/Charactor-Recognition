import time
import torch


def localtime():
    '''
    Get current time
    '''
    return time.strftime('%Y%m%d%H%M%S', time.localtime())


def save(net, state_dict=False, replace=False):
    '''
    Save a network
    '''
    if replace:
        if state_dict:
            torch.save(net.state_dict(),
                       f'./saved_nets/best_state.pkl')
        else:
            torch.save(net, f'./saved_nets/best.pkl')
    else:
        if state_dict:
            torch.save(net.state_dict(),
                       f'./saved_nets/net_state_{localtime()}.pkl')
        else:
            torch.save(net, f'./saved_nets/net_{localtime()}.pkl')


def restore(pkl_path, model_class=None):
    '''
    Restore a network
    '''
    if model_class != None:
        try:
            model = model_class()
            return model.load_state_dict(torch.load(pkl_path))
        except:
            raise ValueError(
                'model_class must match with the model you want to restore')

    else:
        return torch.load(pkl_path)
