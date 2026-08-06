"""
run.py
======
Launcher - ap dung _compat.py (va albumentations) TRUOC khi chay script
that (tt_sfuda_2d.py hoac tt_sfuda_2d_dualema.py), khong sua bat ky dong
nao trong 2 file do.

Cach dung (thay the hoan toan cho `python tt_sfuda_2d.py ...`):

    python run.py tt_sfuda_2d.py --source chase_unet --target hrf
    python run.py tt_sfuda_2d_dualema.py --source chase_unet --target hrf --topology parallel
"""
import sys
import runpy

import _compat  # noqa: F401 - PHAI import truoc, tu dong va sys.modules

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Cach dung: python run.py <ten_script.py> [cac --tham-so khac]")
        sys.exit(1)

    target_script = sys.argv[1]
    sys.argv = sys.argv[1:]   # de argparse trong script dich khong thay ten run.py
    runpy.run_path(target_script, run_name='__main__')
