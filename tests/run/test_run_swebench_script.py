import subprocess


def test_help_describes_automatic_system_sampling():
    result = subprocess.run(["bash", "run-swebench.sh", "--help"], check=True, capture_output=True, text=True)

    assert "--system-metrics-interval SECONDS" in result.stdout
    assert "eight-GPU metrics" in result.stdout


def test_instance_selector_rejects_regex_as_exact_id():
    result = subprocess.run(
        ["bash", "run-swebench.sh", "--instance", "^(django__django-11099)$"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "Invalid exact instance ID" in result.stderr
    assert "django__django-11099" in result.stderr
    assert "use --filter for regex" in result.stderr
