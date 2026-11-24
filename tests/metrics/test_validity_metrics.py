"""Tests for validity metrics implementation."""

import json
import math
import time
import warnings
from collections import defaultdict
from itertools import combinations_with_replacement, product
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from pymatgen.core import Composition, Element, Lattice, Species
from pymatgen.core.structure import Structure
from pymatgen.util.testing import PymatgenTest

from lemat_genbench.metrics import validity_metrics
from lemat_genbench.metrics.validity_metrics import (
    ChargeNeutralityMetric,
    MinimumInteratomicDistanceMetric,
    OverallValidityMetric,
    PhysicalPlausibilityMetric,
)
from lemat_genbench.utils.oxidation_state import (
    _bounded_oxi_combos,
    _iter_target_sum_tuples,
    compositional_oxi_state_guesses,
    electronegativity_correlation,
)


@pytest.mark.parametrize(
    ("elements", "oxidation_states"),
    [
        (["Cu", "Si"], [1, -1]),
        (["Li", "F"], [1, 1]),
        (["Cu"], [1]),
    ],
)
def test_electronegativity_correlation_constant_vector(elements, oxidation_states):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        assert electronegativity_correlation(elements, oxidation_states) == 0.0


def test_electronegativity_correlation_nonconstant_vector():
    assert electronegativity_correlation(["Ni", "P"], [1, -1]) < 0.0


def test_equal_electronegativity_charge_fallback(monkeypatch):
    structure = Structure(
        Lattice.cubic(10), ["Cu", "Si"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )
    actual_guesses = compositional_oxi_state_guesses
    fallback_calls = []

    def guesses(comp, *, all_oxi_states, **kwargs):
        if not all_oxi_states:
            return (), (), ()
        fallback_calls.append(comp)
        return actual_guesses(comp, all_oxi_states=all_oxi_states, **kwargs)

    monkeypatch.setattr(validity_metrics, "metallicity_score", lambda _: 0.0)
    monkeypatch.setattr(
        validity_metrics, "get_inequivalent_site_info", Mock(side_effect=ValueError)
    )
    monkeypatch.setattr(validity_metrics, "compositional_oxi_state_guesses", guesses)
    analyzer = Mock()
    analyzer.get_oxi_state_decorated_structure.side_effect = ValueError

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        deviation = ChargeNeutralityMetric.compute_structure(
            structure, bv_analyzer=analyzer
        )

    assert fallback_calls == [Composition("CuSi")]
    assert deviation == 0.0


def _brute_force_oxi_guesses(
    comp: Composition,
    *,
    all_oxi_states: bool,
    oxi_states_override: dict[str, list] | None,
    target_charge: float = 0,
) -> tuple[tuple, tuple, tuple]:
    """Reference implementation matching the original brute-force algorithm."""
    comp = comp.reduced_composition
    data_dir = Path(__file__).resolve().parents[2] / "src" / "lemat_genbench" / "data"
    with open(data_dir / "lemat_icsd_oxi_dict_probs.json", "r") as f:
        oxi_prob = json.load(f)

    oxi_states_override = oxi_states_override or {}
    el_amt = comp.get_el_amt_dict()
    n_sites = int(sum(el_amt.values()))
    elements = list(el_amt)
    el_sums: list = []
    el_sum_scores: defaultdict = defaultdict(dict)
    el_best_oxid_combo: dict = {}

    for idx, el in enumerate(elements):
        el_sum_scores[idx] = {}
        el_best_oxid_combo[idx] = {}
        el_sums.append([])
        if oxi_states_override.get(el):
            oxids: list | tuple = oxi_states_override[el]
        elif all_oxi_states:
            oxids = Element(el).oxidation_states
        else:
            oxids = (
                Element(el).icsd_oxidation_states or Element(el).common_oxidation_states
            )

        for oxid_combo in combinations_with_replacement(oxids, int(el_amt[el])):
            if max(oxid_combo) - min(oxid_combo) <= 1:
                oxid_sum = sum(oxid_combo)
                if oxid_sum not in el_sums[idx]:
                    el_sums[idx].append(oxid_sum)
                if not all_oxi_states:
                    score = math.prod(oxi_prob[str(Species(el, o))] for o in oxid_combo)
                    if (
                        oxid_sum not in el_sum_scores[idx]
                        or score > el_sum_scores[idx][oxid_sum]
                    ):
                        el_sum_scores[idx][oxid_sum] = score
                        el_best_oxid_combo[idx][oxid_sum] = oxid_combo

    el_sums = [[x for x in sublist if x != 0] for sublist in el_sums]
    all_sols, all_oxid_combo, all_scores = [], [], []
    for x in product(*el_sums):
        if sum(x) == target_charge:
            sol = {el: v / el_amt[el] for el, v in zip(elements, x, strict=True)}
            all_sols.append(sol)
            if not all_oxi_states:
                all_scores.append(
                    math.prod(el_sum_scores[idx][v] for idx, v in enumerate(x))
                    ** (1 / n_sites)
                )
                all_oxid_combo.append(
                    {
                        e: el_best_oxid_combo[idx][v]
                        for idx, (e, v) in enumerate(zip(elements, x, strict=True))
                    }
                )
            else:
                all_scores.append(
                    electronegativity_correlation(
                        elements=list(sol.keys()), oxidation_states=list(sol.values())
                    )
                )

    if not all_scores:
        return (), (), ()
    if all_oxi_states:
        sorted_data = sorted(zip(all_scores, all_sols), key=lambda item: item[0])
        all_scores, all_sols = zip(*sorted_data)
        return tuple(all_sols), tuple(all_sols), tuple(all_scores)
    sorted_data = sorted(
        zip(all_scores, all_sols, all_oxid_combo), key=lambda item: item[0], reverse=True
    )
    all_scores, all_sols, all_oxid_combo = zip(*sorted_data)
    return tuple(all_sols), tuple(all_oxid_combo), tuple(all_scores)


@pytest.mark.parametrize(
    ("oxids", "n_atoms"),
    [
        ([-4, -3, -2, -1, 1, 2, 3, 4], 5),
        ([-2, 0, 2], 3),
        ([3, 1, 2, 2], 4),
        ([1], 4),
        ([], 3),
        ([1, 2, 3], 1),
        ([1, 2, 3], 0),
        ([1.5, 2.0, 2.5], 3),
    ],
)
def test_bounded_oxi_combos_matches_brute_force(oxids, n_atoms):
    expected = {
        tuple(sorted(combo))
        for combo in combinations_with_replacement(oxids, n_atoms)
        if not combo or max(combo) - min(combo) <= 1
    }
    actual = [tuple(sorted(combo)) for combo in _bounded_oxi_combos(oxids, n_atoms)]
    assert len(actual) == len(set(actual))
    assert set(actual) == expected


@pytest.mark.parametrize(
    ("el_sums", "target"),
    [
        ([[1, 2, -1]], 2),
        ([[1, 2], [-1, -2]], 0),
        ([[1, 2, 3], [-4, -2, 0], [1, 2, 3], [-3, -1]], 0),
        ([[1, 2], []], 0),
        ([[2, 4], [2, 4]], 1),
        ([[1.5, 2.5], [-1.5, -0.5]], 1.0),
        ([[1, 1, 2], [-1, -2, -1]], 0),
    ],
)
def test_iter_target_sum_tuples_matches_product(el_sums, target):
    expected = [x for x in product(*el_sums) if sum(x) == target]
    actual = list(_iter_target_sum_tuples(el_sums, target))
    assert actual == expected


@pytest.mark.parametrize(
    "formula",
    [
        "Fe3O4",
        "LiFePO4",
        "CuSi",
        "NaCl",
        "SrTiO3",
        "Mn2O3",
        "Ba2Ca4As3H2",
        "YBa2Cu3O7",
    ],
)
def test_compositional_oxi_state_guesses_matches_brute_force(formula):
    data_dir = Path(__file__).resolve().parents[2] / "src" / "lemat_genbench" / "data"
    with open(data_dir / "lemat_icsd_oxi_state_mapping.json", "r") as f:
        mapping = json.load(f)

    comp = Composition(formula)
    override = {str(el): mapping[str(el)] for el in comp.elements if str(el) in mapping}

    for all_oxi, ov in [(False, override), (True, None)]:
        actual = compositional_oxi_state_guesses(
            comp,
            all_oxi_states=all_oxi,
            max_sites=-1,
            oxi_states_override=ov,
            target_charge=0,
        )
        expected = _brute_force_oxi_guesses(
            comp, all_oxi_states=all_oxi, oxi_states_override=ov, target_charge=0
        )
        assert actual[2] == pytest.approx(expected[2])
        assert {tuple(sorted(sol.items())) for sol in actual[0]} == {
            tuple(sorted(sol.items())) for sol in expected[0]
        }
        if not all_oxi:
            actual_combos = {
                tuple((el, tuple(sorted(vals))) for el, vals in sorted(combo.items()))
                for combo in actual[1]
            }
            expected_combos = {
                tuple((el, tuple(sorted(vals))) for el, vals in sorted(combo.items()))
                for combo in expected[1]
            }
            assert actual_combos == expected_combos


def test_cu7si25_oxi_state_guesses():
    data_dir = Path(__file__).resolve().parents[2] / "src" / "lemat_genbench" / "data"
    with open(data_dir / "lemat_icsd_oxi_state_mapping.json", "r") as f:
        mapping = json.load(f)

    comp = Composition("Cu7Si25")
    override = {str(el): mapping[str(el)] for el in comp.elements if str(el) in mapping}
    icsd_res = compositional_oxi_state_guesses(
        comp,
        all_oxi_states=False,
        max_sites=-1,
        oxi_states_override=override,
        target_charge=0,
    )
    assert icsd_res == ((), (), ())

    all_res = compositional_oxi_state_guesses(
        comp,
        all_oxi_states=True,
        max_sites=-1,
        oxi_states_override=None,
        target_charge=0,
    )
    assert len(all_res[0]) > 0
    assert all_res[2][0] == 0.0
    for sol in all_res[0]:
        assert abs(sol["Cu"] * 7 + sol["Si"] * 25) < 1e-9


def test_large_site_performance_and_correctness():
    data_dir = Path(__file__).resolve().parents[2] / "src" / "lemat_genbench" / "data"
    with open(data_dir / "lemat_icsd_oxi_state_mapping.json", "r") as f:
        mapping = json.load(f)

    # CuSi64 (64 Si sites, irreducibly large) has no charge-balanced solution
    # because 64 * 1 > max(Cu) = 4, yet previously enumerated C(71, 7) combos.
    comp_unbalanced = Composition("CuSi64")
    override_unbalanced = {
        str(el): mapping[str(el)]
        for el in comp_unbalanced.elements
        if str(el) in mapping
    }
    start = time.perf_counter()
    icsd_res = compositional_oxi_state_guesses(
        comp_unbalanced,
        all_oxi_states=False,
        max_sites=-1,
        oxi_states_override=override_unbalanced,
        target_charge=0,
    )
    all_res = compositional_oxi_state_guesses(
        comp_unbalanced,
        all_oxi_states=True,
        max_sites=-1,
        oxi_states_override=None,
        target_charge=0,
    )
    elapsed_unbalanced = time.perf_counter() - start
    assert elapsed_unbalanced < 1.0
    assert icsd_res == ((), (), ())
    assert all_res == ((), (), ())

    # Cu17Si64 (coprime, 64 reduced Si sites) has charge-balanced solutions.
    comp_balanced = Composition("Cu17Si64")
    override_balanced = {
        str(el): mapping[str(el)]
        for el in comp_balanced.elements
        if str(el) in mapping
    }
    start = time.perf_counter()
    icsd_balanced = compositional_oxi_state_guesses(
        comp_balanced,
        all_oxi_states=False,
        max_sites=-1,
        oxi_states_override=override_balanced,
        target_charge=0,
    )
    all_balanced = compositional_oxi_state_guesses(
        comp_balanced,
        all_oxi_states=True,
        max_sites=-1,
        oxi_states_override=None,
        target_charge=0,
    )
    elapsed_balanced = time.perf_counter() - start

    assert elapsed_balanced < 1.0
    assert isinstance(icsd_balanced, tuple) and len(icsd_balanced) == 3
    assert len(all_balanced[0]) > 0
    assert all_balanced[2][0] == 0.0
    for sol in all_balanced[0]:
        assert abs(sol["Cu"] * 17 + sol["Si"] * 64) < 1e-9


def test_multielement_large_site_target_sum_performance():
    comp = Composition("Fe4Mn4Cr4O14")
    el_amt = comp.get_el_amt_dict()

    start = time.perf_counter()
    res = compositional_oxi_state_guesses(
        comp,
        all_oxi_states=True,
        max_sites=None,
        oxi_states_override=None,
        target_charge=0,
    )
    elapsed = time.perf_counter() - start

    assert elapsed < 1.0
    assert len(res[0]) > 0
    assert list(res[2]) == sorted(res[2])
    for sol in res[0]:
        assert abs(sum(sol[el] * amt for el, amt in el_amt.items())) < 1e-9


@pytest.fixture
def valid_structures():
    """Create valid test structures."""
    test = PymatgenTest()
    structures = [
        test.get_structure("Si"),  # Silicon
        test.get_structure("LiFePO4"),  # Lithium iron phosphate
        test.get_structure("CsCl"),  # Cesium chloride
    ]
    return structures


@pytest.fixture
def invalid_structures():
    """Create structures with validity issues."""
    # Create Si structure with extremely compressed lattice
    test = PymatgenTest()
    si = test.get_structure("Si")

    # Create compressed structure - use proper method to scale the lattice
    compressed_lattice = si.lattice.scale(0.1)
    compressed_si = Structure(compressed_lattice, si.species, si.frac_coords)

    # Create structure with atoms too close
    overlapping_si = si.copy()
    for i in range(len(overlapping_si)):
        if i > 0:
            displacement = overlapping_si[i].coords - overlapping_si[0].coords
            displacement /= np.linalg.norm(displacement)
            overlapping_si[i].coords = overlapping_si[0].coords + displacement * 0.5
            break

    return [compressed_si, overlapping_si]


def test_charge_neutrality_metric(valid_structures):
    """Test ChargeNeutralityMetric on valid structures."""
    metric = ChargeNeutralityMetric()
    result = metric.compute(valid_structures)

    # Check computation didn't fail
    assert len(result.failed_indices) == 0

    # Check result structure - updated metric names
    assert "charge_neutral_ratio" in result.metrics
    assert "charge_neutral_count" in result.metrics
    assert "avg_charge_deviation" in result.metrics
    assert "total_structures" in result.metrics
    assert result.primary_metric == "charge_neutral_ratio"

    # Check values
    assert 0.0 <= result.metrics["charge_neutral_ratio"] <= 1.0
    assert result.metrics["charge_neutral_count"] <= result.metrics["total_structures"]
    assert result.metrics["total_structures"] == len(valid_structures)

    # Check one value per structure
    assert len(result.individual_values) == len(valid_structures)
    
    # Individual values should be charge deviations (not binary)
    assert all(v >= 0.0 for v in result.individual_values if not np.isnan(v))


def test_minimum_interatomic_distance_metric(valid_structures, invalid_structures):
    """Test MinimumInteratomicDistanceMetric."""
    metric = MinimumInteratomicDistanceMetric()

    # Test on valid structures
    valid_result = metric.compute(valid_structures)
    
    # Check updated metric names
    assert "distance_valid_ratio" in valid_result.metrics
    assert "distance_valid_count" in valid_result.metrics
    assert "total_structures" in valid_result.metrics
    assert valid_result.primary_metric == "distance_valid_ratio"
    
    # Check values
    assert 0.0 <= valid_result.metrics["distance_valid_ratio"] <= 1.0
    assert valid_result.metrics["total_structures"] == len(valid_structures)
    
    # Individual values should be binary (0.0 or 1.0)
    assert all(v in [0.0, 1.0] for v in valid_result.individual_values if not np.isnan(v))

    # Test on structure with overlapping atoms
    invalid_result = metric.compute([invalid_structures[1]])  # The overlapping structure

    # Check that overlapping structure gets lower score
    assert (
        invalid_result.metrics["distance_valid_ratio"]
        <= valid_result.metrics["distance_valid_ratio"]
    )


def test_physical_plausibility_metric(valid_structures, invalid_structures):
    """Test PhysicalPlausibilityMetric."""
    metric = PhysicalPlausibilityMetric()

    # Test on valid structures
    valid_result = metric.compute(valid_structures)
    
    # Check updated metric names
    assert "plausibility_valid_ratio" in valid_result.metrics
    assert "plausibility_valid_count" in valid_result.metrics
    assert "total_structures" in valid_result.metrics
    assert valid_result.primary_metric == "plausibility_valid_ratio"
    
    # Check values
    assert 0.0 <= valid_result.metrics["plausibility_valid_ratio"] <= 1.0
    assert valid_result.metrics["total_structures"] == len(valid_structures)
    
    # Individual values should be binary (0.0 or 1.0)
    assert all(v in [0.0, 1.0] for v in valid_result.individual_values if not np.isnan(v))

    # Test on structure with compressed lattice (unrealistic density)
    invalid_result = metric.compute([invalid_structures[0]])  # The compressed structure

    # Check that compressed structure gets lower score
    assert (
        invalid_result.metrics["plausibility_valid_ratio"]
        <= valid_result.metrics["plausibility_valid_ratio"]
    )


def test_overall_validity_metric(valid_structures, invalid_structures):
    """Test OverallValidityMetric requires ALL checks to pass."""
    metric = OverallValidityMetric(
        charge_tolerance=0.1,
        distance_scaling=0.5,
        min_atomic_density=0.00001,
        max_atomic_density=0.5,
        min_mass_density=0.01,
        max_mass_density=25.0,
        check_format=False,  # Skip for speed
        check_symmetry=False,
    )

    # Test on valid structures
    valid_result = metric.compute(valid_structures)
    
    # Check result structure
    assert "overall_valid_ratio" in valid_result.metrics
    assert "overall_valid_count" in valid_result.metrics
    assert "total_structures" in valid_result.metrics
    assert valid_result.primary_metric == "overall_valid_ratio"
    
    # Check values
    assert 0.0 <= valid_result.metrics["overall_valid_ratio"] <= 1.0
    assert valid_result.metrics["total_structures"] == len(valid_structures)
    
    # Individual values should be binary (0.0 or 1.0)
    assert all(v in [0.0, 1.0] for v in valid_result.individual_values if not np.isnan(v))

    # Test on invalid structures
    invalid_result = metric.compute(invalid_structures)

    # Valid structures should have higher or equal overall validity
    assert (
        valid_result.metrics["overall_valid_ratio"]
        >= invalid_result.metrics["overall_valid_ratio"]
    )


def test_charge_neutrality_tolerance():
    """Test ChargeNeutralityMetric with different tolerance."""
    # Create a structure with slight charge imbalance
    test = PymatgenTest()
    structure = test.get_structure("LiFePO4")

    # Create metrics with different tolerances
    strict_metric = ChargeNeutralityMetric(tolerance=0.01)
    lenient_metric = ChargeNeutralityMetric(tolerance=1.0)

    # Evaluate structure
    strict_result = strict_metric.compute([structure])
    lenient_result = lenient_metric.compute([structure])

    # Lenient metric should classify more structures as charge neutral
    assert (
        lenient_result.metrics["charge_neutral_ratio"]
        >= strict_result.metrics["charge_neutral_ratio"]
    )


def test_interatomic_distance_scaling():
    """Test MinimumInteratomicDistanceMetric with different scaling factors."""
    test = PymatgenTest()
    structure = test.get_structure("Si")

    # Create metrics with different scaling factors
    strict_metric = MinimumInteratomicDistanceMetric(scaling_factor=0.9)
    lenient_metric = MinimumInteratomicDistanceMetric(scaling_factor=0.3)

    # Evaluate structure
    strict_result = strict_metric.compute([structure])
    lenient_result = lenient_metric.compute([structure])

    # Lenient metric should allow atoms to be closer
    assert (
        lenient_result.metrics["distance_valid_ratio"]
        >= strict_result.metrics["distance_valid_ratio"]
    )


def test_charge_neutrality_deviation_tracking():
    """Test that ChargeNeutralityMetric tracks deviations properly."""
    test = PymatgenTest()
    structures = [test.get_structure("Si")] * 3
    
    metric = ChargeNeutralityMetric(tolerance=0.1)
    result = metric.compute(structures)
    
    # Should have deviation information
    assert "avg_charge_deviation" in result.metrics
    assert not np.isnan(result.metrics["avg_charge_deviation"]) or result.metrics["avg_charge_deviation"] >= 0
    
    # Individual values should be charge deviations
    assert len(result.individual_values) == len(structures)
    assert all(isinstance(v, (int, float)) for v in result.individual_values if not np.isnan(v))


def test_overall_validity_intersection_logic():
    """Test that overall validity is intersection of all individual checks."""
    test = PymatgenTest()
    valid_si = test.get_structure("Si")
    
    # Create structure that might fail only one check
    compressed_si = Structure(
        valid_si.lattice.scale(0.1), 
        valid_si.species, 
        valid_si.frac_coords
    )
    
    structures = [valid_si, compressed_si]
    
    # Test individual metrics
    charge_metric = ChargeNeutralityMetric(tolerance=0.1)
    distance_metric = MinimumInteratomicDistanceMetric(scaling_factor=0.5)
    plausibility_metric = PhysicalPlausibilityMetric(check_format=False, check_symmetry=False)
    overall_metric = OverallValidityMetric(
        charge_tolerance=0.1,
        distance_scaling=0.5,
        check_format=False,
        check_symmetry=False
    )
    
    charge_result = charge_metric.compute(structures)
    distance_result = distance_metric.compute(structures)
    plausibility_result = plausibility_metric.compute(structures)
    overall_result = overall_metric.compute(structures)
    
    # Overall count should be <= min of individual counts
    overall_count = overall_result.metrics["overall_valid_count"]
    charge_count = charge_result.metrics["charge_neutral_count"]
    distance_count = distance_result.metrics["distance_valid_count"]
    plausibility_count = plausibility_result.metrics["plausibility_valid_count"]
    
    assert overall_count <= min(charge_count, distance_count, plausibility_count)


def test_count_ratio_consistency():
    """Test that counts and ratios are mathematically consistent."""
    test = PymatgenTest()
    structures = [test.get_structure("Si")] * 5  # 5 identical structures
    
    metric = ChargeNeutralityMetric()
    result = metric.compute(structures)
    
    total = result.metrics["total_structures"]
    count = result.metrics["charge_neutral_count"]
    ratio = result.metrics["charge_neutral_ratio"]
    
    assert total == 5
    assert abs(ratio - (count / total)) < 1e-10


def test_physical_plausibility_checks():
    """Test individual checks in PhysicalPlausibilityMetric."""
    test = PymatgenTest()
    structure = test.get_structure("Si")
    
    # Test with different check configurations
    density_only = PhysicalPlausibilityMetric(
        check_format=False, 
        check_symmetry=False
    )
    
    all_checks = PhysicalPlausibilityMetric(
        check_format=True,
        check_symmetry=True
    )
    
    density_result = density_only.compute([structure])
    all_result = all_checks.compute([structure])
    
    # Both should work, but all_checks might be more restrictive
    assert 0.0 <= density_result.metrics["plausibility_valid_ratio"] <= 1.0
    assert 0.0 <= all_result.metrics["plausibility_valid_ratio"] <= 1.0
    assert all_result.metrics["plausibility_valid_ratio"] <= density_result.metrics["plausibility_valid_ratio"]


def test_overall_validity_parameters():
    """Test OverallValidityMetric with different parameter sets."""
    test = PymatgenTest()
    structure = test.get_structure("Si")
    
    # Strict parameters
    strict_metric = OverallValidityMetric(
        charge_tolerance=0.001,
        distance_scaling=0.9,
        min_atomic_density=0.01,
        max_atomic_density=0.2,
        min_mass_density=2.0,
        max_mass_density=20.0,
        check_format=True,
        check_symmetry=True
    )
    
    # Lenient parameters
    lenient_metric = OverallValidityMetric(
        charge_tolerance=1.0,
        distance_scaling=0.1,
        min_atomic_density=0.00001,
        max_atomic_density=0.5,
        min_mass_density=0.1,
        max_mass_density=50.0,
        check_format=False,
        check_symmetry=False
    )
    
    strict_result = strict_metric.compute([structure])
    lenient_result = lenient_metric.compute([structure])
    
    # Lenient should have higher or equal validity
    assert (lenient_result.metrics["overall_valid_ratio"] >= 
            strict_result.metrics["overall_valid_ratio"])
