"""Pair-consistent training utilities for factorized graph heads."""
import torch

ORIENTATION_SWAP=(1,0,2)


def symmetrize_factorized_components(skeleton,orientation):
    """Return pair-symmetric skeleton and swap-equivariant orientation logits."""
    if skeleton.ndim!=3 or orientation.shape!=(*skeleton.shape,3):
        raise ValueError("Expected B x N x N skeleton and B x N x N x 3 orientation")
    if skeleton.shape[1]!=skeleton.shape[2]:raise ValueError("Expected square pair axes")
    skel=.5*(skeleton+skeleton.transpose(1,2))
    reverse=orientation.transpose(1,2)[...,ORIENTATION_SWAP]
    orient=.5*(orientation+reverse)
    return skel,orient
