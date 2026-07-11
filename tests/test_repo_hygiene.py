import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_repo_hygiene.py"


def init_repo(path: Path) -> None:
    subprocess.run(["git", "init"], cwd=path, check=True, stdout=subprocess.PIPE)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)


def run_checker(path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(path)],
        cwd=path,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def test_detects_secret_and_private_path(tmp_path: Path) -> None:
    init_repo(tmp_path)
    sample = tmp_path / "README.md"
    sample.write_text(
        "api_key = \"abcdefghijklmnopqrstuvwxyz123456\"\n"
        "bad path /export/home/alice/project\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "README.md"], cwd=tmp_path, check=True)

    result = run_checker(tmp_path)

    assert result.returncode == 1
    assert "generic API key assignment" in result.stdout
    assert "internal absolute path /export/home" in result.stdout


def test_detects_generated_assets_and_missing_submodule(tmp_path: Path) -> None:
    init_repo(tmp_path)
    (tmp_path / "output.png").write_bytes(b"not really an image")
    subprocess.run(["git", "add", "output.png"], cwd=tmp_path, check=True)

    result = run_checker(tmp_path)

    assert result.returncode == 1
    assert "test generated image should not be tracked" in result.stdout
    assert "missing submodule" in result.stdout


def test_detects_broken_markdown_link(tmp_path: Path) -> None:
    init_repo(tmp_path)
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "guide.md").write_text("[missing](missing.md)\n", encoding="utf-8")
    subprocess.run(["git", "add", "docs/guide.md"], cwd=tmp_path, check=True)

    result = run_checker(tmp_path)

    assert result.returncode == 1
    assert "broken relative Markdown link" in result.stdout
