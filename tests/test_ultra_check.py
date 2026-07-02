import ultra_check


def test_readiness_config_check_passes():
    result = ultra_check._check_config()

    assert result["passed"] is True


def test_readiness_run_command_reports_failure():
    result = ultra_check._run("intentional_failure", ["python", "-c", "raise SystemExit(7)"])

    assert result["passed"] is False
    assert result["returncode"] == 7


def test_readiness_code_graph_smoke_is_nonempty():
    result = ultra_check._check_code_graph_smoke()

    assert result["passed"] is True
    assert '"nodes"' in result["output"]


def test_readiness_package_metadata_passes():
    result = ultra_check._check_package_metadata()

    assert result["passed"] is True
