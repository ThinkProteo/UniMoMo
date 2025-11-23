#!/usr/bin/python
# -*- coding:utf-8 -*-

from .EPT.ept import XTransEncoderAct as EPT
from .EPT.ept import XTransEncoderActMoT as EPTMoT

def create_net(
    name,
    hidden_size,
    edge_size,
    opt={}
):
    if name == 'EPT':
        kargs = {
            'hidden_size': hidden_size,
            'ffn_size': hidden_size,
            'edge_size': edge_size
        }
        kargs.update(opt)
        return EPT(**kargs)
    elif name == 'EPTMoT':
        kargs = {
            'hidden_size': hidden_size,
            'ffn_size': hidden_size,
            'edge_size': edge_size
        }
        kargs.update(opt)
        return EPTMoT(**kargs)
    else:
        raise NotImplementedError(f'{name} not implemented')
