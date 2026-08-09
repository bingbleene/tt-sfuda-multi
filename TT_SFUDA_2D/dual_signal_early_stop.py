"""
dual_signal_early_stop.py
============================
NANG CAP UnsupervisedEarlyStopper (khong sua file cu - de A/B so sanh dam
bao, giong cach lam voi tt_sfuda_2d_control_cldice.py vs tt_sfuda_2d_
region_topology.py truoc day).

LY DO / CO SO (dua tren toan bo du lieu da thu thap trong du an):
- 3 thi nghiem tim "nguon thong tin doc lap moi" (cross-arch bottleneck,
  wavelet, SAM) DEU that bai o buoc validate doc lap hoac do phan ky.
  Ket luan: Dual-EMA (2 teacher CUNG kien truc, khac keep_rate) KHONG co
  ensemble gain thuc su ve CHAT LUONG pseudo-label - da chung minh bang
  disagreement ~0.002-0.003 (qua nho de mang thong tin moi).
- NHUNG ket qua tot nhat du an (58.20 +/- 0.09, CHASE->HRF) den tu
  UnsupervisedEarlyStopper + rollback, KHONG phai tu ensemble. Co che nay
  hien CHI dung 1 tin hieu (ty le pixel duong tinh cua tgt_model).
- Huong nang cap nay KHONG co gang bien fast/slow thanh nguon thong tin
  moi (da chung minh khong duoc). Thay vao do, KHAI THAC DUNG BAN CHAT
  CUA NO: fast-EMA (keep_rate thap hon, vd 0.99) phan ung voi thay doi
  gan day cua student NHANH HON slow-EMA (keep_rate cao hon, vd 0.995).
  Khi student bat dau troi dat/collapse, fast va slow se bat dau LECH
  NHAU truoc khi ty le pixel duong tinh cua tgt_model (chi 1 model, do
  RIENG) kip vuot nguong ratio_change_threshold hien tai. Day la vai tro
  "canh bao som" (early-warning), KHONG phai "nguon thong tin doc lap" -
  2 vai tro khac nhau, va vai tro nay CHUA tung duoc thu.

QUAN TRONG - day la GIA THUYET CO CO SO, CHUA duoc kiem chung thuc te:
can chay A/B that (ben cu vs ben nay, cung seed, cung domain shift) va
so sanh SO LAN rollback kich hoat DUNG LUC (truoc khi Dice thuc su sup,
neu theo doi rieng de doi chieu) truoc khi ket luan day la cai tien that.
KHONG mac dinh tin no hoat dong chi vi co ly do nghe hop ly - dung bai
hoc "luon kiem tra bang debug so lieu cu the truoc khi tin ket qua".
"""
import copy
from collections import deque

import torch

from unsupervised_early_stop import ImageOnlyDataset  # tai dung, khong viet lai


class DualSignalEarlyStopper:
    def __init__(self, warmup_epochs=3, window_size=5, unstable_count_threshold=3,
                 ratio_change_threshold=0.20, disagreement_change_threshold=0.20):
        """
        Giu NGUYEN toan bo logic warmup/window/threshold cua ban cu cho tin
        hieu "ratio" (xem unsupervised_early_stop.py de doi chieu tung dong).
        THEM tin hieu "disagreement" (chenh lech fast/slow) song song, CUNG
        co che baseline-co-dinh-sau-warmup + nguong-thay-doi-tuong-doi.

        disagreement_change_threshold: MAC DINH giong ratio_change_threshold
        (0.20) VI CHUA CO DU LIEU THUC TE de biet ty le hop ly rieng cho tin
        hieu nay - CAN calibrate lai sau khi xem gia tri thuc te in ra qua
        vai lan chay dau (giong cach threshold 0.20 cua ratio duoc chon tu
        dau - khong doan mo mai, nhung cung khong gia dinh dung ngay).
        """
        self.warmup_epochs = warmup_epochs
        self.window_size = window_size
        self.unstable_count_threshold = unstable_count_threshold
        self.ratio_change_threshold = ratio_change_threshold
        self.disagreement_change_threshold = disagreement_change_threshold

        self.ratio_history = []
        self.disagreement_history = []
        self.ratio_baseline = None
        self.disagreement_baseline = None
        self.recent_stability = deque(maxlen=window_size)

        self.best_state_dict = None
        self.best_epoch = -1

    @torch.no_grad()
    def _predicted_positive_ratio(self, model, image_only_loader):
        model.eval()
        total_ratio, n = 0.0, 0
        for input in image_only_loader:
            input = input.cuda()
            output = model(input)
            prob = torch.sigmoid(output)
            ratio = (prob > 0.5).float().mean().item()
            total_ratio += ratio * input.size(0)
            n += input.size(0)
        model.train()
        return total_ratio / n

    @torch.no_grad()
    def _fast_slow_disagreement(self, teacher_manager, image_only_loader):
        """Chenh lech TRUNG BINH |prob_fast - prob_slow| tren toan tap anh -
        CHI ho tro teacher_manager co dung 2 teacher ten 'fast'/'slow'
        (dung quy uoc dat ten da co san trong multi_teacher.py)."""
        assert 'fast' in teacher_manager.teachers and 'slow' in teacher_manager.teachers, (
            "DualSignalEarlyStopper can teacher_manager co ca 'fast' va 'slow' "
            "(dung dung ten teacher nhu cau hinh --topology parallel hien co).")
        teacher_manager.train_mode(False)
        total_diff, n = 0.0, 0
        for input in image_only_loader:
            input = input.cuda()
            prob_fast = torch.sigmoid(teacher_manager.teachers['fast'](input))
            prob_slow = torch.sigmoid(teacher_manager.teachers['slow'](input))
            diff = (prob_fast - prob_slow).abs().mean().item()
            total_diff += diff * input.size(0)
            n += input.size(0)
        return total_diff / n

    def check(self, model, teacher_manager, image_only_loader, epoch_idx, verbose=True):
        """Goi sau MOI epoch, TRUYEN THEM teacher_manager so voi ban cu.
        Tra ve True neu nen DUNG training tai day."""
        ratio = self._predicted_positive_ratio(model, image_only_loader)
        disagreement = self._fast_slow_disagreement(teacher_manager, image_only_loader)
        self.ratio_history.append(ratio)
        self.disagreement_history.append(disagreement)

        if len(self.ratio_history) <= self.warmup_epochs:
            self.best_state_dict = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch_idx
            if len(self.ratio_history) == self.warmup_epochs:
                self.ratio_baseline = self._median(self.ratio_history)
                self.disagreement_baseline = self._median(self.disagreement_history)
            if verbose:
                print(f"  [DualSignal-EarlyStop] epoch {epoch_idx}: ty_le={ratio:.4f}, "
                      f"chenh_lech_fast_slow={disagreement:.5f} "
                      f"(warmup {len(self.ratio_history)}/{self.warmup_epochs})")
            return False

        ratio_change = abs(ratio - self.ratio_baseline) / (self.ratio_baseline + 1e-6)
        disagreement_change = abs(disagreement - self.disagreement_baseline) / (self.disagreement_baseline + 1e-6)

        ratio_unstable = ratio_change > self.ratio_change_threshold
        disagreement_unstable = disagreement_change > self.disagreement_change_threshold
        # OR logic: BAT KY tin hieu nao bao dong deu tinh la epoch bat on -
        # muc tieu la BAT SOM HON, chap nhan co the tang false positive so
        # voi ban cu (chi dung 1 tin hieu) - can theo doi thuc te co dung
        # tang false-stop qua muc khong khi so sanh A/B.
        is_unstable = ratio_unstable or disagreement_unstable
        is_stable = not is_unstable
        self.recent_stability.append(is_stable)

        if is_stable:
            self.best_state_dict = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch_idx

        unstable_in_window = sum(1 for s in self.recent_stability if not s)

        if verbose:
            status = 'ON DINH' if is_stable else 'BAT ON'
            trigger = []
            if ratio_unstable:
                trigger.append('ty_le')
            if disagreement_unstable:
                trigger.append('chenh_lech_fast_slow')
            trigger_str = f" [kich hoat boi: {', '.join(trigger)}]" if trigger else ""
            print(f"  [DualSignal-EarlyStop] epoch {epoch_idx}: ty_le={ratio:.4f} "
                  f"(lech {ratio_change*100:.1f}%), chenh_lech_fast_slow={disagreement:.5f} "
                  f"(lech {disagreement_change*100:.1f}%) -> {status}{trigger_str} "
                  f"(bat on {unstable_in_window}/{len(self.recent_stability)} trong {self.window_size} epoch gan nhat)")

        return (len(self.recent_stability) == self.window_size and
                unstable_in_window >= self.unstable_count_threshold)

    @staticmethod
    def _median(values):
        s = sorted(values)
        mid = len(s) // 2
        return s[mid] if len(s) % 2 == 1 else (s[mid - 1] + s[mid]) / 2

    def restore_best(self, model):
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)
        return self.best_epoch
