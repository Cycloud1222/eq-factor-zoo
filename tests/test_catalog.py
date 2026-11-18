from factor_zoo.catalog import FACTOR_CATALOG, FACTOR_NAMES


def test_catalog_has_24_unique_factors() -> None:
    assert len(FACTOR_CATALOG) == 24
    assert len(FACTOR_NAMES) == 24
    assert len(set(FACTOR_NAMES)) == 24


def test_every_factor_has_a_formula_and_family() -> None:
    assert all(spec.formula for spec in FACTOR_CATALOG)
    assert all(spec.family for spec in FACTOR_CATALOG)
