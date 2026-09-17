import os
import subprocess


def test_installer_is_idempotent_without_ripgrep(tmp_path):
    env_root = tmp_path / "env"
    site_packages = env_root / "site-packages"
    target = site_packages / "vllm/v1/engine/output_processor.py"
    target.parent.mkdir(parents=True)
    target.write_text(
        "from vllm.request_metrics_jsonl import log_finished_request_metrics\n"
        "        log_finished_request_metrics(request_id, finish_reason, prompt_tokens, stats)\n"
    )

    python = env_root / "bin/python"
    python.parent.mkdir()
    python.write_text(
        "#!/bin/sh\n"
        "case \"$2\" in\n"
        "  *importlib.metadata*) printf '%s\\n' '0.10.1.1' ;;\n"
        f"  *'import site'*) printf '%s\\n' '{site_packages}' ;;\n"
        "esac\n"
    )
    python.chmod(0o755)

    environment = os.environ | {"PATH": "/usr/bin:/bin"}
    subprocess.run(
        ["bash", "scripts/install-vllm-instrumentation.sh", str(env_root)],
        check=True,
        env=environment,
    )

    assert not target.with_suffix(".py.rej").exists()
    assert (site_packages / "vllm/request_metrics_jsonl.py").is_file()
