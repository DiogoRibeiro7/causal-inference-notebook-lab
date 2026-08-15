"""Utilities for a notebook-first causal inference portfolio project."""

# Single source of the version. pyproject.toml reads this attribute, and
# tests/test_metadata.py asserts CITATION.cff and .zenodo.json agree with it.
__version__ = "0.3.0"

from causal_inference_lab.data_generators import (
    SyntheticDataset,
    make_confounded_binary_treatment,
    make_did_panel,
    make_heterogeneous_treatment_data,
    make_iv_data,
)
from causal_inference_lab.difference_in_differences import (
    DifferenceInDifferencesResult,
    difference_in_differences,
)
from causal_inference_lab.dml import double_machine_learning_ate
from causal_inference_lab.estimators import (
    aipw_ate,
    difference_in_means,
    g_computation_ate,
    ipw_ate,
)
from causal_inference_lab.instrumental_variables import IVResult, instrumental_variables_ate
from causal_inference_lab.matching import (
    MatchingResult,
    matching_balance_table,
    nearest_neighbour_matching,
    propensity_score_matching,
)
from causal_inference_lab.meta_learners import (
    CATEModel,
    SMetaLearner,
    TMetaLearner,
    XMetaLearner,
)
from causal_inference_lab.rdd import (
    RDDResult,
    local_linear_rdd,
    rdd_bandwidth_sensitivity,
)
from causal_inference_lab.reporting import CausalReport
from causal_inference_lab.synthetic_control import SyntheticControlResult, fit_synthetic_control
from causal_inference_lab.uncertainty import BootstrapResult, bootstrap_ate

__all__ = [
    "__version__",
    "SyntheticDataset",
    "make_confounded_binary_treatment",
    "make_heterogeneous_treatment_data",
    "make_did_panel",
    "make_iv_data",
    "difference_in_means",
    "g_computation_ate",
    "ipw_ate",
    "aipw_ate",
    "double_machine_learning_ate",
    "MatchingResult",
    "matching_balance_table",
    "nearest_neighbour_matching",
    "propensity_score_matching",
    "bootstrap_ate",
    "BootstrapResult",
    "CATEModel",
    "SMetaLearner",
    "TMetaLearner",
    "XMetaLearner",
    "DifferenceInDifferencesResult",
    "difference_in_differences",
    "IVResult",
    "instrumental_variables_ate",
    "RDDResult",
    "local_linear_rdd",
    "rdd_bandwidth_sensitivity",
    "CausalReport",
    "SyntheticControlResult",
    "fit_synthetic_control",
]
