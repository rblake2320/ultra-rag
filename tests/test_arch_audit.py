import ultra_arch_audit


def test_architecture_audit_passes_contract_checks():
    result = ultra_arch_audit.run_audit()

    assert result["passed"] is True


def test_architecture_audit_requires_failure_runbook():
    assert "docs/FAILURE_RUNBOOK.md" in ultra_arch_audit.REQUIRED_FILES


def test_architecture_audit_forbids_unmeasured_superiority_claims():
    assert "best in class" in ultra_arch_audit.FORBIDDEN_CLAIMS
