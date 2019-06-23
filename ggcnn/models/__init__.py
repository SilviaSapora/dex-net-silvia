def get_network(network_name):
    network_name = network_name.lower()
    if network_name == 'ggcnn':
        from .ggcnn import GGCNN
        return GGCNN
    elif network_name == 'ggcnn2':
        from .ggcnn2 import GGCNN2
        return GGCNN2
    elif network_name == 'ggcnn3':
        from .ggcnn3 import GGCNN3
        return GGCNN3
    elif network_name == 'ggcnn4':
        from .ggcnn4 import GGCNN4
        return GGCNN4
    elif network_name == 'ggcnn5':
        from .ggcnn5 import GGCNN5
        return GGCNN5
    else:
        raise NotImplementedError('Network {} is not implemented'.format(network_name))
