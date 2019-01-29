#!/usr/bin/env python
# coding: utf-8

# # Introduction

# This notebook presents couple examples of **graph visualization** in **pytorch**.

# **NOTE**: graph vizualization code was originally copied from https://github.com/szagoruyko/pytorchviz. Subsequenty modified by me.

# Contents:
# * [Imports](#Imports)
# * [Graph Vizualization](#Graph-Vizualization)
# * [Simple Operations Test](#Simple-Operations-Test)
# * [Simple Model Test](#Simple-Model-Test)
# * [U-Net Test](#U-Net-Test)
# * [ResNet18 Test](#ResNet18-Test)

# # Imports

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F


# Pick GPU if available

# In[3]:


device = 'cpu' # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# # Graph Vizualization

# Functions in this section are copy-pasted from [https://github.com/szagoruyko/pytorchviz](https://github.com/szagoruyko/pytorchviz)

# In[4]:


from graphviz import Digraph


# In[5]:


class GradFnHook():
    def __init__(self, grad_fn):
        self.iter = 0
        self.grad_fn = grad_fn
    def __call__(self, inputs, outputs):
        self.iter += 1
        self.inputs = inputs
        self.outputs = outputs


# In[6]:


def add_hooks(*output_tensors):
    def add_nodes(var):
        if var not in seen:
            
            bh = GradFnHook(var)
            var.register_hook(bh)
            
            hooks[var] = bh
            seen.add(var)
            
            #if hasattr(var, 'next_functions'):
            for u in var.next_functions:
                if u[0] is not None:
                    add_nodes(u[0])

    # convert to nodes
    nodes = tuple(ot.grad_fn for ot in output_tensors)
    
    hooks = {}
    seen = set()
    
    for grad_fn in nodes:
        add_nodes(grad_fn)
        
    return hooks


# In[7]:


def size_to_str(size):
    if len(size) == 0:
        return '(scalar)'
    return '(' + (', ').join(['%d' % v for v in size]) + ')'


# In[8]:


def var_to_str(var):
    def rchop(string, endings):
        for ending in endings:
            if string.endswith(ending):
                return string[:-len(ending)]
        return string
    return rchop(type(var).__name__, ['Backward', 'Backward0', 'Backward1', 'Backward2', 'Backward3'])


# In[9]:


class DummyGradFn():
    def __init__(self, tensor):
        self.variable = tensor
        self.next_functions = [[tensor.grad_fn, 0]]


# In[10]:


def make_dot(outputs, inputs=None, params=None, hooks=None):
    """ Produces Graphviz representation of PyTorch autograd graph.
    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function
    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """

    def add_nodes(var):
        if var not in seen:
            if hasattr(var, 'variable'):
                tensor = var.variable
                
                if id(tensor) in output_map:
                    name = output_map[id(tensor)]
                    color = 'darkolivegreen1'
                elif id(tensor) in param_map:
                    name = param_map[id(tensor)]
                    color = 'orange'
                elif id(tensor) in input_map:
                    name = input_map[id(tensor)]
                    color = 'lightblue'
                else:
                    name = ''
                    color = 'lightgray'
            
                node_name = str.format('{}\n{}', name, size_to_str(tensor.size()))
                dot.node(str(id(var)), node_name, fillcolor=color, style='filled', shape='box')
            
            else:
                
                if hooks is not None and var in hooks:
                    output_tensors = hooks[var].outputs
                    node_name = var_to_str(var)
                    for tensor in output_tensors:
                        node_name += '\n' + size_to_str(tensor.size())
                    
                else:
                    node_name = var_to_str(var)
                
                dot.node(str(id(var)), node_name, fillcolor='lightgray')
            
            seen.add(var)
            
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        label = None
#                         if hooks is not None and var in hooks and u[0] in hooks:
#                             inp_tup = hooks[var].inputs  # tensor
#                             out_tup = hooks[u[0]].outputs
#                             inters = set(inp_tup).intersection(set(out_tup))
#                             if len(inters) == 1:  # if multiple paths through graph then this will fail
#                                 tensor = inters.pop()
#                                 label = size_to_str(tensor.size())
                        
                        dot.edge(str(id(u[0])), str(id(var)), label=label)
                        add_nodes(u[0])

    assert all(isinstance(o, torch.Tensor) for o in outputs.values())
    output_map = {id(tensor): string for string, tensor in outputs.items()}
        
    if inputs is not None:
        assert all(isinstance(i, torch.Tensor) for i in inputs.values())
        input_map = {id(tensor): string for string, tensor in inputs.items()}
    else:
        input_map = {}
        
    if params is not None:
        assert all(isinstance(p, torch.Tensor) for p in params.values())
        param_map = {id(tensor): string for string, tensor in params.items()}
    else:
        param_map = {}

    node_attr = dict(style='rounded,filled', shape='box', align='top', fontsize='12', ranksep='0.1', height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="10,1000"))

    seen = set()

    output_nodes = [DummyGradFn(tensor) for _, tensor in outputs.items()]
            
            
    for grad_fn in output_nodes:
        add_nodes(grad_fn)


    return dot


