"""
multi_teacher.py (v2)
======================
Mo rong tu ban v1: giu nguyen 2 topology (parallel/cascaded) va EMA update,
THEM 3 che do ket hop pseudo-label o topology='parallel':

  - 'mean'       : trung binh deu (HANH VI CU, mac dinh, tuong thich nguoc)
  - 'weighted'   : trung binh co trong so CO DINH (vd fast=0.7, slow=0.3),
                    trong so lay tu key 'weight' trong teacher_configs
  - 'confidence' : trong so DONG theo tung PIXEL, dua tren entropy - teacher
                    nao it "phan van" hon (entropy thap hon) o vi tri do duoc
                    tin tuong hon. Y tuong nay dua truc tiep tu chinh ky thuat
                    "selective voting" cua paper goc (Eq. 5-7), nhung ap dung
                    giua CAC TEACHER thay vi giua cac augmentation.

topology='cascaded' giu nguyen (dung output teacher cuoi chuoi) - ensemble_mode
khong anh huong toi cascaded.
"""
import copy
from collections import OrderedDict

import torch


class MultiTeacherManager:
    def __init__(self, teacher_configs, topology='parallel'):
        """
        teacher_configs: list[dict], vd:
            [{'name': 'fast', 'keep_rate': 0.99, 'weight': 0.7},
             {'name': 'slow', 'keep_rate': 0.999, 'weight': 0.3}]
        'weight' la TUY CHON - chi can khi dung ensemble_mode='weighted'.
        Neu khong co, mac dinh weight=1.0 (tro thanh trung binh deu khi
        normalize).
        """
        assert topology in ('parallel', 'cascaded'), \
            f"topology phai la 'parallel' hoac 'cascaded', nhan duoc: {topology}"
        assert len(teacher_configs) >= 1, "Can it nhat 1 teacher"

        self.topology = topology
        self.order = [c['name'] for c in teacher_configs]
        self.keep_rates = {c['name']: c['keep_rate'] for c in teacher_configs}
        self.static_weights = {c['name']: c.get('weight', 1.0) for c in teacher_configs}
        self.teachers = {}
        self._progress = 1.0   # dung cho ensemble_mode='warmup', xem set_progress()

    def set_progress(self, progress):
        """
        progress trong [0,1] - ty le epoch da hoan thanh trong Stage II.
        CHI anh huong ensemble_mode='warmup'. Goi 1 lan moi epoch (khong
        phai moi iteration) tu vong lap huan luyen ben ngoai.
        """
        self._progress = max(0.0, min(1.0, progress))

    def init_from(self, base_model):
        base_state = base_model.state_dict()
        for name in self.order:
            teacher = copy.deepcopy(base_model)
            teacher.load_state_dict(base_state)
            teacher.eval()
            self.teachers[name] = teacher
        return self

    @staticmethod
    def _ema_update(teacher_model, source_model, keep_rate):
        source_dict = source_model.state_dict()
        new_dict = OrderedDict()
        for key, value in teacher_model.state_dict().items():
            if key in source_dict:
                new_dict[key] = source_dict[key] * (1 - keep_rate) + value * keep_rate
            else:
                raise Exception(f"'{key}' khong tim thay trong source model")
        teacher_model.load_state_dict(new_dict)

    @torch.no_grad()
    def update(self, student_model):
        if self.topology == 'parallel':
            for name in self.order:
                self._ema_update(self.teachers[name], student_model, self.keep_rates[name])
        else:  # cascaded
            source = student_model
            for name in self.order:
                self._ema_update(self.teachers[name], source, self.keep_rates[name])
                source = self.teachers[name]

    @staticmethod
    def _entropy(prob, eps=1e-8):
        return -(prob * torch.log(prob + eps) + (1 - prob) * torch.log(1 - prob + eps))

    @torch.no_grad()
    def predict(self, input_tensor, mode=None, ensemble_mode='mean'):
        """
        ensemble_mode chi co tac dung khi topology='parallel'.
        Tra ve: (prob, feats) neu mode='const', nguoc lai chi tra ve prob.
        """
        if self.topology == 'cascaded':
            last_name = self.order[-1]
            if mode == 'const':
                out, feats = self.teachers[last_name](input_tensor, mode='const')
                return torch.sigmoid(out), feats
            out = self.teachers[last_name](input_tensor)
            return torch.sigmoid(out)

        # ==== topology == 'parallel' ====
        assert ensemble_mode in ('mean', 'weighted', 'confidence', 'warmup', 'trust_region'), \
            f"ensemble_mode khong hop le: {ensemble_mode}"

        outs = {}   # name -> prob
        feats_per_teacher = {}
        for name in self.order:
            if mode == 'const':
                out, feats = self.teachers[name](input_tensor, mode='const')
                feats_per_teacher[name] = feats
            else:
                out = self.teachers[name](input_tensor)
            outs[name] = torch.sigmoid(out)

        if ensemble_mode == 'mean':
            weights = {name: 1.0 / len(self.order) for name in self.order}

        elif ensemble_mode == 'weighted':
            total = sum(self.static_weights[name] for name in self.order)
            weights = {name: self.static_weights[name] / total for name in self.order}

        elif ensemble_mode == 'warmup':
            # Chi ho tro dung 2 teacher ten 'fast'/'slow'. Trong so 'slow'
            # TANG DAN theo tien do Stage II (self._progress, cap nhat moi
            # epoch qua set_progress()): dau Stage II gan nhu chi nghe
            # 'fast' (giong baseline, tu do thich nghi), cuoi Stage II moi
            # nghe 'slow' theo dung ty le dich (static_weights['slow']).
            # Muc dich: tranh teacher 'slow' "neo" vao checkpoint Stage I
            # con kem ngay tu dau (nguyen nhan da chan doan gay hai o RITE).
            assert set(self.order) == {'fast', 'slow'}, \
                "ensemble_mode='warmup' chi ho tro dung 2 teacher ten 'fast' va 'slow'"
            total_w = self.static_weights['fast'] + self.static_weights['slow']
            target_slow_w = self.static_weights['slow'] / total_w  # vd 1.0/1.0 -> 0.5 (mac dinh)
            slow_w = self._progress * target_slow_w
            weights = {'fast': 1.0 - slow_w, 'slow': slow_w}

        elif ensemble_mode == 'trust_region':
            raise ValueError(
                "ensemble_mode='trust_region' phai goi qua predict_with_trust(), "
                "khong phai predict() - vi can tra ve them trust_mask."
            )

        else:  # 'confidence' - trong so DONG theo tung pixel, dua tren entropy
            eps = 1e-6
            inv_ent = {name: 1.0 / (self._entropy(outs[name]) + eps) for name in self.order}
            total_inv = sum(inv_ent[name] for name in self.order)
            weights = {name: inv_ent[name] / total_inv for name in self.order}

        avg_prob = sum(weights[name] * outs[name] for name in self.order)

        if mode == 'const':
            # feature dung cho consistency loss KHONG lien quan pseudo-label
            # -> luon trung binh deu, bat ke ensemble_mode (feature khong co
            # khai niem "do tin cay" nhu xac suat)
            n_layers = len(feats_per_teacher[self.order[0]])
            avg_feats = [
                sum(feats_per_teacher[name][i] for name in self.order) / len(self.order)
                for i in range(n_layers)
            ]
            return avg_prob, avg_feats
        return avg_prob

    @torch.no_grad()
    def predict_with_trust(self, input_tensor, agree_threshold=0.5, mode='const'):
        """
        'trust_region': KHONG gop 2 xac suat thanh 1 con so trung gian. Thay
        vao do:
          - agree_mask = 1 o nhung pixel ca 'fast' va 'slow' CUNG dong y ve
            nhan nhi phan (ca hai >=0.5 hoac ca hai <0.5), 0 o nhung pixel
            bat dong.
          - pseudo_label lay tu 'fast' (thich nghi nhanh hon, uu tien khi
            CA HAI DA DONG Y - luc do dung ai cung nhu nhau).
          - O vung bat dong, KHONG ep student hoc theo ben nao ca (mask=0
            -> loai khoi loss hoan toan), thay vi thoa hiep (trung binh)
            nhu cac ensemble_mode khac - day la diem khac biet cot loi.

        Chi ho tro dung 2 teacher 'fast'/'slow', topology='parallel'.
        Tra ve: (pseudo_label, trust_mask, feats)
        """
        assert self.topology == 'parallel', "trust_region chi ho tro topology='parallel'"
        assert set(self.order) == {'fast', 'slow'}, \
            "trust_region chi ho tro dung 2 teacher ten 'fast' va 'slow'"

        out_fast, feats_fast = self.teachers['fast'](input_tensor, mode='const')
        out_slow, feats_slow = self.teachers['slow'](input_tensor, mode='const')
        prob_fast = torch.sigmoid(out_fast)
        prob_slow = torch.sigmoid(out_slow)

        label_fast = (prob_fast >= agree_threshold).float()
        label_slow = (prob_slow >= agree_threshold).float()
        trust_mask = (label_fast == label_slow).float()

        pseudo_label = label_fast  # o vung dong y, label_fast == label_slow, chon ben nao cung nhu nhau

        n_layers = len(feats_fast)
        avg_feats = [(feats_fast[i] + feats_slow[i]) / 2 for i in range(n_layers)]

        return pseudo_label, trust_mask, avg_feats

    def train_mode(self, flag=False):
        for m in self.teachers.values():
            m.train(flag)
    def cuda(self):
        for m in self.teachers.values():
            m.cuda()
        return self

    def state_dicts(self):
        return {name: m.state_dict() for name, m in self.teachers.items()}

    def load_state_dicts(self, state_dicts):
        for name, sd in state_dicts.items():
            self.teachers[name].load_state_dict(sd)
