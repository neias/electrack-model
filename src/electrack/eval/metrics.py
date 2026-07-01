"""Değerlendirme metrikleri — saf Python (US2: SC-001, SC-002, SC-004).

Tanımlar (spec Clarifications / research.md R7):
- overall_recall: mikro-ortalama recall @ IoU 0.5. Bir ground-truth kutu, IoU≥0.5 VE
  doğru sınıflı bir tahminle eşleşirse "yakalandı" sayılır.
- false_positive_rate: gerçek bir komponente (herhangi sınıf, IoU≥0.5) karşılık
  gelmeyen tahminlerin, tüm tahminlere oranı (hayali/phantom tespit — FR-007).
  Not: gerçek bir komponentin üzerindeki 'unknown' tahmini FP DEĞİLDİR (nesne vardır).

Kutular normalize [0-1] xyxy. Tahmin = {'bbox','class_name','class_id','confidence'}.
Ground-truth = {'class_id', 'bbox'(xyxy normalize)}.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from electrack.config_constants import IOU_MATCH_THRESHOLD, UNKNOWN_LABEL


def iou_xyxy(a: list[float], b: list[float]) -> float:
    """İki [x1,y1,x2,y2] kutusunun IoU'su."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class ClassMetric:
    tp: int = 0
    fn: int = 0  # yakalanmayan GT
    pred_total: int = 0  # bu sınıfa ait tahmin sayısı
    pred_correct: int = 0  # doğru-sınıf eşleşen tahmin

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def precision(self) -> float:
        return self.pred_correct / self.pred_total if self.pred_total else 0.0

    @property
    def support(self) -> int:
        return self.tp + self.fn


@dataclass
class ImageEval:
    gts: list[dict]  # {'class_id','bbox'}
    preds: list[dict]  # sözleşme tespitleri


@dataclass
class AggregateMetrics:
    total_gt: int = 0
    matched_gt: int = 0  # doğru-sınıf + IoU≥0.5
    total_pred: int = 0
    phantom_pred: int = 0  # hiçbir GT ile IoU≥0.5 örtüşmeyen
    per_class: dict[int, ClassMetric] = field(default_factory=dict)

    @property
    def overall_recall(self) -> float:
        return self.matched_gt / self.total_gt if self.total_gt else 0.0

    @property
    def false_positive_rate(self) -> float:
        return self.phantom_pred / self.total_pred if self.total_pred else 0.0


def _match_image(
    gts: list[dict],
    preds: list[dict],
    iou_thr: float,
    agg: AggregateMetrics,
) -> None:
    """Tek görüntü için GT↔tahmin eşleştirmesi (greedy, güven azalan sırada)."""
    agg.total_gt += len(gts)
    agg.total_pred += len(preds)

    for g in gts:
        agg.per_class.setdefault(g["class_id"], ClassMetric())

    preds_sorted = sorted(preds, key=lambda p: p["confidence"], reverse=True)
    gt_used_correct = [False] * len(gts)

    for p in preds_sorted:
        pid = p.get("class_id")
        if pid is not None:
            cm = agg.per_class.setdefault(pid, ClassMetric())
            cm.pred_total += 1

        # Phantom mu? Herhangi bir GT ile IoU≥0.5 örtüşmesi var mı?
        overlaps_any = any(iou_xyxy(p["bbox"], g["bbox"]) >= iou_thr for g in gts)
        if not overlaps_any:
            agg.phantom_pred += 1

        # Doğru-sınıf eşleşme (recall için) — yalnızca somut sınıflı tahminler.
        if pid is None or p["class_name"] == UNKNOWN_LABEL:
            continue
        best_j, best_iou = -1, iou_thr
        for j, g in enumerate(gts):
            if gt_used_correct[j] or g["class_id"] != pid:
                continue
            v = iou_xyxy(p["bbox"], g["bbox"])
            if v >= best_iou:
                best_iou, best_j = v, j
        if best_j >= 0:
            gt_used_correct[best_j] = True
            agg.per_class[pid].pred_correct += 1

    # GT sonuçlarını topla.
    for j, g in enumerate(gts):
        cm = agg.per_class[g["class_id"]]
        if gt_used_correct[j]:
            cm.tp += 1
            agg.matched_gt += 1
        else:
            cm.fn += 1


def evaluate(images: list[ImageEval], iou_thr: float = IOU_MATCH_THRESHOLD) -> AggregateMetrics:
    agg = AggregateMetrics()
    for im in images:
        _match_image(im.gts, im.preds, iou_thr, agg)
    return agg


def per_class_report(agg: AggregateMetrics, id_to_name) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for cid, cm in sorted(agg.per_class.items()):
        name = id_to_name(cid) if callable(id_to_name) else id_to_name[cid]
        out[name] = {
            "recall": round(cm.recall, 6),
            "precision": round(cm.precision, 6),
            "support": cm.support,
        }
    return out


# --------------------------- Unknown davranışı (SC-004) --------------------------- #


def unknown_behavior(
    ambiguous_images: list[ImageEval],
    class_threshold: float,
    iou_thr: float = IOU_MATCH_THRESHOLD,
) -> dict:
    """Belirsiz nesneli görüntülerde davranış.

    - high_conf_misclassifications: belirsiz bölgede somut sınıflı + güven≥class_threshold
      tahmin (istenmeyen; hedef düşük/0).
    - unknown_recall: belirsiz nesnelerin 'unknown' olarak yakalanma oranı.

    Belirsiz görüntülerde `gts`, belirsiz nesnelerin kutularını (class_id=-1) taşır.
    """
    high_conf_mis = 0
    ambiguous_total = 0
    ambiguous_caught = 0
    for im in ambiguous_images:
        ambiguous_total += len(im.gts)
        caught = [False] * len(im.gts)
        for p in im.preds:
            overlaps = [
                j for j, g in enumerate(im.gts) if iou_xyxy(p["bbox"], g["bbox"]) >= iou_thr
            ]
            if not overlaps:
                continue
            if p["class_name"] == UNKNOWN_LABEL:
                for j in overlaps:
                    caught[j] = True
            elif p["confidence"] >= class_threshold:
                high_conf_mis += 1
        ambiguous_caught += sum(caught)
    return {
        "high_conf_misclassifications": high_conf_mis,
        "unknown_recall": round(ambiguous_caught / ambiguous_total, 6) if ambiguous_total else 0.0,
    }
