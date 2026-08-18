"""Small interactive terminal UI for full-context NLA AV probes.

Run this from a VSCode Remote SSH terminal on AutoDL:

    cd /root/autodl-tmp/natural_language_autoencoders
    /root/miniconda3/bin/python demo/answer_probe_tui.py

The UI intentionally exposes only the knobs that are useful during exploration:
prompt, AR scoring, extraction layer, answer temperature, AV temperature, and
output path. Everything else uses the Qwen/NLA defaults from the local AutoDL setup.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable
ANSWER_PROBE = ROOT / "demo" / "answer_probe.py"

BASE_MODEL = "/root/autodl-tmp/models/Qwen2.5-7B-Instruct"
AV_MODEL = "/root/autodl-tmp/models/nla-qwen-av"
AR_MODEL = "/root/autodl-tmp/models/nla-qwen-ar"
SGLANG_URL = "http://localhost:30000"
OUT_DIR = Path("/root/autodl-tmp")


@dataclass
class Settings:
    prompt: str = "Explain why the Eiffel Tower is famous in two sentences."
    use_ar: bool = True
    answer_temperature: float = 0.0
    av_temperature: float = 0.0
    layer_index: int = 20
    max_answer_tokens: int = 256
    out_path: Path = OUT_DIR / "full_context_av.json"


def clear() -> None:
    print("\033[2J\033[H", end="")


def pause() -> None:
    input("\nPress Enter to continue...")


def prompt_bool(current: bool) -> bool:
    default = "y" if current else "n"
    raw = input(f"Enable AR scoring? [y/n] ({default}): ").strip().lower()
    if not raw:
        return current
    return raw in {"y", "yes", "1", "true", "on"}


def prompt_float(label: str, current: float) -> float:
    raw = input(f"{label} ({current}): ").strip()
    if not raw:
        return current
    try:
        return float(raw)
    except ValueError:
        print(f"Invalid float: {raw!r}")
        pause()
        return current


def prompt_int(label: str, current: int) -> int:
    raw = input(f"{label} ({current}): ").strip()
    if not raw:
        return current
    try:
        value = int(raw)
    except ValueError:
        print(f"Invalid integer: {raw!r}")
        pause()
        return current
    return max(1, value)


def prompt_path(current: Path) -> Path:
    raw = input(f"Output JSON path ({current}): ").strip()
    if not raw:
        return current
    return Path(raw).expanduser()


def prompt_text(current: str) -> str:
    print("Enter prompt. For multiple lines, enter /multi and finish with a single '.' line.")
    print(f"Current: {current}")
    first = input("> ")
    if not first:
        return current
    if first.strip() != "/multi":
        return first.replace("\\n", "\n")

    lines: list[str] = []
    while True:
        line = input()
        if line == ".":
            break
        lines.append(line)
    text = "\n".join(lines).strip()
    return text or current


def check_av_server() -> tuple[bool, str]:
    try:
        with urllib.request.urlopen(f"{SGLANG_URL}/get_model_info", timeout=3) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        return True, body[:160]
    except (OSError, urllib.error.URLError) as exc:
        return False, str(exc)


def tail(path: Path, lines: int = 40) -> str:
    if not path.exists():
        return ""
    data = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(data[-lines:])


def build_command(settings: Settings) -> list[str]:
    cmd = [
        PYTHON,
        str(ANSWER_PROBE),
        "--base-model", BASE_MODEL,
        "--av", AV_MODEL,
        "--prompt", settings.prompt,
        "--target", "all",
        "--positions", "all",
        "--quiet",
        "--layer-index", str(settings.layer_index),
        "--max-answer-tokens", str(settings.max_answer_tokens),
        "--answer-temperature", str(settings.answer_temperature),
        "--av-temperature", str(settings.av_temperature),
        "--out", str(settings.out_path),
    ]
    if settings.use_ar:
        cmd[cmd.index("--prompt"):cmd.index("--prompt")] = ["--ar", AR_MODEL]
    return cmd


def run_probe(settings: Settings) -> None:
    ok, info = check_av_server()
    if not ok:
        print("AV server does not look reachable at http://localhost:30000.")
        print(info)
        print("\nStart it in another terminal:")
        print("  AV_DIR=/root/autodl-tmp/models/nla-qwen-av MEM_FRACTION=0.6 bash demo/launch_av_server.sh")
        pause()
        return

    settings.out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = settings.out_path.with_suffix(settings.out_path.suffix + ".log")
    cmd = build_command(settings)

    print("AV server: OK")
    print("Running probe. Long model logs are going to:")
    print(f"  {log_path}")
    print("JSON result will be written to:")
    print(f"  {settings.out_path}")

    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    elapsed = time.time() - started

    if proc.returncode != 0:
        print(f"\nRun failed with exit code {proc.returncode}. Log tail:")
        print(tail(log_path))
        pause()
        return

    print(f"\nDone in {elapsed:.1f}s.")
    try:
        payload = json.loads(settings.out_path.read_text(encoding="utf-8"))
        print(f"Answer tokens: {payload.get('answer_token_count')}")
        print(f"Probe rows: {len(payload.get('results', []))}")
        print(f"AR scoring: {'on' if settings.use_ar else 'off'}")
    except Exception as exc:  # noqa: BLE001 - diagnostic UI
        print(f"Could not summarize JSON: {exc}")
    pause()


def show_settings(settings: Settings) -> None:
    clear()
    print("NLA Full-Context AV Probe")
    print("=" * 72)
    print(f"1. Prompt             : {settings.prompt}")
    print(f"2. AR scoring         : {'on' if settings.use_ar else 'off'}")
    print(f"3. Answer temperature : {settings.answer_temperature}")
    print(f"4. AV temperature     : {settings.av_temperature}")
    print(f"5. Layer index        : {settings.layer_index}")
    print(f"6. Max answer tokens  : {settings.max_answer_tokens}")
    print(f"7. Output JSON        : {settings.out_path}")
    print("-" * 72)
    print("Fixed: target=all, positions=all, quiet export")
    if settings.layer_index != 20:
        print("NOTE: AV/AR checkpoints are Qwen L20; non-20 layers are intentional mismatch probes.")
    print("8. Run")
    print("9. Check AV server")
    print("q. Quit")


def main() -> None:
    settings = Settings()
    while True:
        show_settings(settings)
        choice = input("\nSelect: ").strip().lower()
        if choice == "1":
            settings.prompt = prompt_text(settings.prompt)
            stem = time.strftime("full_context_av_%Y%m%d_%H%M%S")
            settings.out_path = OUT_DIR / f"{stem}.json"
        elif choice == "2":
            settings.use_ar = prompt_bool(settings.use_ar)
        elif choice == "3":
            settings.answer_temperature = prompt_float("Answer temperature", settings.answer_temperature)
        elif choice == "4":
            settings.av_temperature = prompt_float("AV temperature", settings.av_temperature)
        elif choice == "5":
            settings.layer_index = prompt_int("Layer index", settings.layer_index)
        elif choice == "6":
            settings.max_answer_tokens = prompt_int("Max answer tokens", settings.max_answer_tokens)
        elif choice == "7":
            settings.out_path = prompt_path(settings.out_path)
        elif choice == "8":
            run_probe(settings)
        elif choice == "9":
            ok, info = check_av_server()
            print("AV server OK." if ok else "AV server not reachable.")
            print(info)
            pause()
        elif choice in {"q", "quit", "exit"}:
            return
        else:
            print(f"Unknown choice: {choice!r}")
            pause()


if __name__ == "__main__":
    main()
