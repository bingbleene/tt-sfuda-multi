"""Flatten BraTS2019 Training (HGG+LGG) into the layout expected by the author's code and
write the 266/69 split.

  out/all/<case> -> symlink to <brats_root>/{HGG,LGG}/<case>
  out/all/all.txt (335)   out/all/val_s<seed>.txt (69 each)   out/all/train266_s<seed>.txt (266)

NOTE: all.txt is the full 335 (author's config `train_list: all.txt`: the unlabeled-target
adaptation set), val.txt is the held-out 69 used for scoring. The author's exact split is not
public, so it is regenerated with a fixed seed.
"""
import argparse, os, random

ap = argparse.ArgumentParser()
ap.add_argument("--brats_root", required=True, help=".../MICCAI_BraTS_2019_Data_Training")
ap.add_argument("--out", default="/kaggle/working/data/train")
ap.add_argument("--seeds", default="2001,1,2,3,4", help="comma separated; one 69-case val list per seed (val_s<seed>.txt). "
                "The author's split is unknown, so several random splits are generated to see how sensitive the metrics are.")
ap.add_argument("--n_val", type=int, default=69)
a = ap.parse_args()

cases = []
for g in ("HGG", "LGG"):
    gd = os.path.join(a.brats_root, g)
    for c in sorted(os.listdir(gd)):
        if os.path.isdir(os.path.join(gd, c)):
            cases.append((c, os.path.join(gd, c)))
assert len(cases) == len({c for c, _ in cases}), "duplicate case ids"
print("found", len(cases), "cases")

dst = os.path.join(a.out, "all")
os.makedirs(dst, exist_ok=True)
for c, p in cases:
    link = os.path.join(dst, c)
    if os.path.lexists(link):
        os.remove(link)
    os.symlink(p, link)

names = sorted(c for c, _ in cases)
def w(fn, lst):
    with open(os.path.join(dst, fn), "w") as f:
        f.write("\n".join(lst) + "\n")

w("all.txt", names)                       # 335: adaptation set + split-independent evaluation
seeds = [int(x) for x in a.seeds.split(",")]
for i, sd in enumerate(seeds):
    sh = names[:]
    random.Random(sd).shuffle(sh)
    val, train = sorted(sh[: a.n_val]), sorted(sh[a.n_val:])
    w(f"val_s{sd}.txt", val)
    w(f"train266_s{sd}.txt", train)
    if i == 0:
        w("val.txt", val)                  # default list of the author's script
print(f"all={len(names)}  val lists: {[f'val_s{x}.txt' for x in seeds]} -> {dst}")