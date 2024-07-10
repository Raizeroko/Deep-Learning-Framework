from net.ACRNN import ACRNN


def choose_net(params):
    if params['net'] == "ACRNN":
        net = ACRNN(params['num_electrodes'])

    return net

