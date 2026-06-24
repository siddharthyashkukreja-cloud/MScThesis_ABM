"""Wire the surrogate optimum into globals.CALIBRATED, safely.

Reads theta_stage2 from output/relock/sur_{regime}.json (written by the overnight
run) and rewrites the CALIBRATED = {...} block in model/globals.py. Makes a backup
first and restores it if the rewritten file fails to import, so an overnight run can
never leave globals broken.

  python3 scripts/wire_lock.py            # uses output/relock/sur_*.json
"""
from __future__ import annotations
import os, sys, re, json, shutil, subprocess
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GLOBALS = os.path.join(_ROOT, "model", "globals.py")
KEYS = ["ft_sigma_c", "zi_alpha", "zi_mu", "zi_delta", "p_zi",
        "mt_lambda", "mt_gamma"]   # write order (7-d loop: p_zi/mt_lambda/mt_gamma now calibrated)


def theta_for(regime):
    p = os.path.join(_ROOT, "output", "relock", f"sur_{regime}.json")
    d = json.load(open(p))["results"][regime]["theta_stage2"]
    return {k: round(float(d[k]), 5) for k in KEYS if k in d}


def block(calm, stressed):
    def fmt(t):
        return ", ".join(f"{k}={t[k]}" for k in KEYS if k in t)
    return ("CALIBRATED = {\n"
            "    # Surrogate (XGBoost) optimum on the D68 widened bounds; auto-wired by\n"
            "    # scripts/wire_lock.py from output/relock/sur_*.json. Grid cross-check deferred.\n"
            f"    \"calm\": dict(\n        {fmt(calm)},\n    ),\n"
            f"    \"stressed\": dict(\n        {fmt(stressed)},\n    ),\n"
            "}")


def main():
    calm, stressed = theta_for("calm"), theta_for("stressed")
    src = open(GLOBALS).read()
    new_block = block(calm, stressed)
    new_src, n = re.subn(r"CALIBRATED = \{.*?\n\}", new_block, src, count=1, flags=re.DOTALL)
    if n != 1:
        sys.exit("ERROR: could not locate the CALIBRATED block in globals.py")
    shutil.copy2(GLOBALS, GLOBALS + ".bak")
    open(GLOBALS, "w").write(new_src)
    chk = subprocess.run([sys.executable, "-c", "from model.globals import CALIBRATED; print(CALIBRATED)"],
                         cwd=_ROOT, capture_output=True, text=True)
    if chk.returncode != 0:
        shutil.copy2(GLOBALS + ".bak", GLOBALS)
        sys.exit(f"ERROR: rewritten globals.py failed to import; restored backup.\n{chk.stderr}")
    print("wired CALIBRATED ->")
    print("  calm    :", calm)
    print("  stressed:", stressed)
    print("(backup at model/globals.py.bak)")


if __name__ == "__main__":
    main()
