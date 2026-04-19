"""Tests for coronalyze.PPConfig."""

import jax

from coronalyze import PPConfig


def test_ppconfig_defaults():
    """PPConfig default field values match the documented defaults."""
    pp = PPConfig()
    assert pp.ppfact == 1.0
    assert pp.n_rolls == 1
    assert pp.ez_ppf == 1.0


def test_ppconfig_custom_fields():
    """PPConfig accepts custom field values via keyword arguments."""
    pp = PPConfig(ppfact=0.5, n_rolls=2, ez_ppf=10.0)
    assert pp.ppfact == 0.5
    assert pp.n_rolls == 2
    assert pp.ez_ppf == 10.0


def test_ppconfig_is_pytree():
    """PPConfig flattens and unflattens losslessly as a JAX pytree."""
    pp = PPConfig(ppfact=0.5, n_rolls=2, ez_ppf=10.0)
    leaves, treedef = jax.tree_util.tree_flatten(pp)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert rebuilt.ppfact == 0.5
    assert rebuilt.n_rolls == 2
    assert rebuilt.ez_ppf == 10.0
