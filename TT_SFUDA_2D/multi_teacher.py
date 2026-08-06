"""
multi_teacher.py
=================
Mo rong TT-SFUDA Stage II (task-specific adaptation) tu 1 teacher (EMA don,
keep_rate=0.99, ham update_teacher_model trong tt_sfuda_2d.py) sang N teacher
(Dual/Multi-EMA Teacher).

KHONG sua bat ky file goc nao (archs.py, dataset.py, losses.py, tt_sfuda_2d.py)
- file nay CHI them, duoc import tu entrypoint rieng (tt_sfuda_2d_dualema.py).

Y tuong:
- Teacher "fast" (keep_rate thap hon, vd 0.99): bam sat student, thich nghi
  nhanh voi target domain nhung de nhieu theo pseudo-label noisy.
- Teacher "slow" (keep_rate cao hon, vd 0.999): thay doi cham, on dinh hon,
  dong vai tro "neo" chong drift khi pseudo-label sai.
- Ket hop 2 (hoac nhieu hon) teacher qua 2 topology:
    * 'parallel'  : moi teacher EMA truc tiep tu student (doc lap nhau),
                    pseudo-label = trung binh xac suat cua tat ca teacher.
    * 'cascaded'  : teacher dau EMA tu student, teacher ke tiep EMA tu
                    CHINH teacher truoc do (chuoi loc muot dan), pseudo-label
                    lay tu teacher CUOI cung trong chuoi (muot nhat).

Cong thuc EMA giu nguyen y het update_teacher_model goc (tt_sfuda_2d.py dong 63-76):
    teacher_new[k] = source[k] * (1 - keep_rate) + teacher_old[k] * keep_rate
"""
import copy
from collections import OrderedDict

import torch


class MultiTeacherManager:
    def __init__(self, teacher_configs, topology='parallel'):
        """
        teacher_configs: list[dict], vd:
            [{'name': 'fast', 'keep_rate': 0.99},
             {'name': 'slow', 'keep_rate': 0.999}]
            Thu tu trong list QUAN TRONG voi topology='cascaded'
            (teacher dau tien nhan EMA truc tiep tu student).
        topology: 'parallel' hoac 'cascaded'
        """
        assert topology in ('parallel', 'cascaded'), \
            f"topology phai la 'parallel' hoac 'cascaded', nhan duoc: {topology}"
        assert len(teacher_configs) >= 1, "Can it nhat 1 teacher"

        self.topology = topology
        self.order = [c['name'] for c in teacher_configs]
        self.keep_rates = {c['name']: c['keep_rate'] for c in teacher_configs}
        self.teachers = {}  # name -> nn.Module, gan trong init_from()

    def init_from(self, base_model):
        """
        Khoi tao tat ca teacher tu CUNG 1 checkpoint (thuong la Theta_t^stage1,
        giong het cach tt_sfuda_2d.py goc khoi tao ca student lan teacher
        tu msrc_model sau Stage I).
        base_model: model DA load checkpoint, cung kien truc archs.UNet.
        """
        base_state = base_model.state_dict()
        for name in self.order:
            teacher = copy.deepcopy(base_model)
            teacher.load_state_dict(base_state)
            teacher.eval()
            self.teachers[name] = teacher
        return self

    @staticmethod
    def _ema_update(teacher_model, source_model, keep_rate):
        """Y het update_teacher_model() goc trong tt_sfuda_2d.py, tach thanh
        staticmethod de dung lai duoc cho ca 2 topology."""
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
        """Goi moi iteration, sau khi student.backward()+optimizer.step(),
        dung dung vi tri nhu update_teacher_model(...) goc trong sfuda_task()."""
        if self.topology == 'parallel':
            for name in self.order:
                self._ema_update(self.teachers[name], student_model, self.keep_rates[name])
        else:  # cascaded
            source = student_model
            for name in self.order:
                self._ema_update(self.teachers[name], source, self.keep_rates[name])
                source = self.teachers[name]

    @torch.no_grad()
    def predict(self, input_tensor, mode=None):
        """
        Sinh pseudo-label + (tuy chon) feature cho consistency loss.

        - 'parallel' : trung binh sigmoid-probability cua TAT CA teacher
                        (giong tinh than 'ensemble' cua Eq.3-4 trong paper,
                        nhung ap dung o Stage II thay vi Stage I).
        - 'cascaded' : chi lay output cua teacher CUOI trong chuoi (teacher
                        da duoc loc muot qua nhieu tang EMA, on dinh nhat).

        Tra ve: (prob_trung_binh_hoac_cuoi, feats) neu mode='const', nguoc lai
        chi tra ve prob.
        """
        if self.topology == 'parallel':
            probs, feats_per_teacher = [], []
            for name in self.order:
                if mode == 'const':
                    out, feats = self.teachers[name](input_tensor, mode='const')
                    feats_per_teacher.append(feats)
                else:
                    out = self.teachers[name](input_tensor)
                probs.append(torch.sigmoid(out))
            avg_prob = sum(probs) / len(probs)

            if mode == 'const':
                n_layers = len(feats_per_teacher[0])
                avg_feats = [
                    sum(f[i] for f in feats_per_teacher) / len(feats_per_teacher)
                    for i in range(n_layers)
                ]
                return avg_prob, avg_feats
            return avg_prob

        else:  # cascaded -> dung teacher cuoi cung (muot nhat)
            last_name = self.order[-1]
            if mode == 'const':
                out, feats = self.teachers[last_name](input_tensor, mode='const')
                return torch.sigmoid(out), feats
            out = self.teachers[last_name](input_tensor)
            return torch.sigmoid(out)

    def train_mode(self, flag=False):
        for m in self.teachers.values():
            m.train(flag)

    def cuda(self):
        for m in self.teachers.values():
            m.cuda()
        return self

    def state_dicts(self):
        """Dung khi can luu checkpoint tat ca teacher (vd de danh gia rieng
        tung teacher, hoac resume training)."""
        return {name: m.state_dict() for name, m in self.teachers.items()}

    def load_state_dicts(self, state_dicts):
        for name, sd in state_dicts.items():
            self.teachers[name].load_state_dict(sd)