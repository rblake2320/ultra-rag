import ultra_product_smoke


def test_product_smoke_passes_static_surface_checks():
    result = ultra_product_smoke.run_smoke()

    assert result["passed"] is True
