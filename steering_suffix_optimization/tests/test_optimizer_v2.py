import torch

from steering_suffix_optimization.controls import hidden_state_index_to_module_index
from steering_suffix_optimization.optimizer import ActivationTarget
from steering_suffix_optimization.optimizer_v2 import ResponsePositionOptimizer


def test_fp32_objective_preserves_sub_bfloat_differences():
    optimizer=object.__new__(ResponsePositionOptimizer)
    optimizer.target=ActivationTarget(4,torch.tensor([1.,0.]),1.)
    optimizer.norm_weight=.1; optimizer.consistency_weight=.25
    one=optimizer.loss_from_deltas(torch.tensor([[1.,0.001]],dtype=torch.float32))
    two=optimizer.loss_from_deltas(torch.tensor([[1.,0.002]],dtype=torch.float32))
    assert one.dtype==torch.float32 and one!=two


def test_hidden_state_to_module_index():
    assert hidden_state_index_to_module_index(4)==3
