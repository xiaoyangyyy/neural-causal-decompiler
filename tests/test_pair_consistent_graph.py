import torch
from ncd.pair_consistent_graph import symmetrize_factorized_components,ORIENTATION_SWAP


def test_factorized_component_symmetrization_is_idempotent_and_equivariant():
    torch.manual_seed(51);s=torch.randn(2,5,5);o=torch.randn(2,5,5,3)
    ss,oo=symmetrize_factorized_components(s,o)
    torch.testing.assert_close(ss,ss.transpose(1,2),rtol=0,atol=0)
    torch.testing.assert_close(oo,oo.transpose(1,2)[...,ORIENTATION_SWAP],rtol=0,atol=0)
    s2,o2=symmetrize_factorized_components(ss,oo)
    torch.testing.assert_close(s2,ss,rtol=0,atol=0);torch.testing.assert_close(o2,oo,rtol=0,atol=0)
