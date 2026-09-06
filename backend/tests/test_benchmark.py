"""WIDER FACE benchmarking helpers (no dataset, no weights)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from app.benchmarking import (
    GtBox,
    average_precision,
    evaluate_detector,
    match_image,
    parse_wider_gt,
    precision_recall_at,
    render_report,
    size_bucket,
)
from app.detectors.base import Detection

# --------------------------------------------------------------------------- #
# GT parsing                                                                   #
# --------------------------------------------------------------------------- #
GT_SAMPLE = """\
0--Parade/0_Parade_marchingband_1_849.jpg
2
449 330 122 149 0 0 0 0 0 0
100 100 40 40 0 0 0 1 0 0
7--Cheering/7_Cheering_Cheering_7_17.jpg
0
0 0 0 0 0 0 0 0 0 0
9--Press_Conference/9_Press_Conference_Press_Conference_9_129.jpg
1
1 1 -5 20 0 0 0 0 0 0
"""


class TestParseWiderGt:
    def test_counts_and_coordinates(self) -> None:
        gt = parse_wider_gt(GT_SAMPLE)
        assert set(gt) == {
            "0--Parade/0_Parade_marchingband_1_849.jpg",
            "7--Cheering/7_Cheering_Cheering_7_17.jpg",
            "9--Press_Conference/9_Press_Conference_Press_Conference_9_129.jpg",
        }
        boxes = gt["0--Parade/0_Parade_marchingband_1_849.jpg"]
        assert len(boxes) == 2
        assert (boxes[0].x1, boxes[0].y1, boxes[0].x2, boxes[0].y2) == (449, 330, 571, 479)

    def test_invalid_flag_parsed(self) -> None:
        boxes = parse_wider_gt(GT_SAMPLE)["0--Parade/0_Parade_marchingband_1_849.jpg"]
        assert boxes[0].invalid is False
        assert boxes[1].invalid is True

    def test_zero_face_block_is_empty_and_consumes_filler(self) -> None:
        gt = parse_wider_gt(GT_SAMPLE)
        assert gt["7--Cheering/7_Cheering_Cheering_7_17.jpg"] == []
        # the block *after* the zero-face one still parses
        assert len(gt["9--Press_Conference/9_Press_Conference_Press_Conference_9_129.jpg"]) == 0

    def test_nonpositive_size_boxes_dropped(self) -> None:
        gt = parse_wider_gt(GT_SAMPLE)
        assert gt["9--Press_Conference/9_Press_Conference_Press_Conference_9_129.jpg"] == []

    def test_trailing_and_blank_lines_ok(self) -> None:
        assert parse_wider_gt("a.jpg\n1\n0 0 10 10 0 0 0 0 0 0\n\n\n")["a.jpg"][0].x2 == 10


class TestSizeBucket:
    @pytest.mark.parametrize(
        ("longest", "bucket"),
        [
            (10, "small"),
            (31, "small"),
            (32, "medium"),
            (96, "medium"),
            (97, "large"),
            (400, "large"),
        ],
    )
    def test_buckets(self, longest: int, bucket: str) -> None:
        assert size_bucket(GtBox(0, 0, longest, longest)) == bucket


# --------------------------------------------------------------------------- #
# Matching                                                                     #
# --------------------------------------------------------------------------- #
class TestMatchImage:
    def test_perfect_match_is_tp(self) -> None:
        gt = GtBox(10, 10, 50, 50)
        m = match_image([Detection(10, 10, 50, 50, 0.9)], [gt], 0.5)
        assert m.scored == [(0.9, True)]
        assert m.n_positives == 1
        assert m.gt_hits == [(size_bucket(gt), True)]

    def test_no_overlap_is_fp_and_fn(self) -> None:
        m = match_image([Detection(0, 0, 5, 5, 0.8)], [GtBox(100, 100, 140, 140)], 0.5)
        assert m.scored == [(0.8, False)]
        assert m.gt_hits == [("medium", False)]

    def test_higher_score_wins_the_single_gt(self) -> None:
        gt = GtBox(0, 0, 40, 40)
        dets = [Detection(0, 0, 40, 40, 0.6), Detection(0, 0, 38, 38, 0.95)]
        m = match_image(dets, [gt], 0.5)
        assert (0.95, True) in m.scored
        assert (0.6, False) in m.scored

    def test_invalid_gt_overlap_is_ignored(self) -> None:
        m = match_image(
            [Detection(0, 0, 40, 40, 0.9)],
            [GtBox(0, 0, 40, 40, invalid=True)],
            0.5,
        )
        assert m.scored == []  # neither TP nor FP
        assert m.n_positives == 0

    def test_iou_below_threshold_is_fp(self) -> None:
        m = match_image([Detection(0, 0, 40, 40, 0.9)], [GtBox(30, 30, 70, 70)], 0.5)
        assert m.scored == [(0.9, False)]


# --------------------------------------------------------------------------- #
# AP / PR                                                                      #
# --------------------------------------------------------------------------- #
class TestAveragePrecision:
    def test_all_true_positives_is_one(self) -> None:
        assert average_precision([(0.9, True), (0.8, True)], 2) == pytest.approx(1.0)

    def test_all_false_positives_is_zero(self) -> None:
        assert average_precision([(0.9, False), (0.8, False)], 2) == pytest.approx(0.0)

    def test_known_mixed_case(self) -> None:
        # TP@.9, FP@.8, TP@.7 with 2 positives -> 0.8333...
        ap = average_precision([(0.9, True), (0.8, False), (0.7, True)], 2)
        assert ap == pytest.approx(0.8333, abs=1e-3)

    def test_no_positives_is_zero(self) -> None:
        assert average_precision([(0.9, False)], 0) == 0.0

    def test_empty_is_zero(self) -> None:
        assert average_precision([], 5) == 0.0


class TestPrecisionRecallAt:
    def test_threshold_filters_low_scores(self) -> None:
        scored = [(0.9, True), (0.6, True), (0.4, False)]
        assert precision_recall_at(scored, 2, 0.5) == (1.0, 1.0)

    def test_nothing_above_threshold(self) -> None:
        assert precision_recall_at([(0.2, True)], 3, 0.5) == (0.0, 0.0)


# --------------------------------------------------------------------------- #
# evaluate_detector / render_report                                            #
# --------------------------------------------------------------------------- #
def _png(path: Path, w: int = 80, h: int = 80) -> None:
    cv2.imwrite(str(path), np.full((h, w, 3), 120, dtype=np.uint8))


class TestEvaluateDetector:
    def test_end_to_end_on_synthetic_samples(self, tmp_path: Path, make_fake_detector) -> None:
        img_a, img_b = tmp_path / "a.png", tmp_path / "b.png"
        _png(img_a)
        _png(img_b)
        gt_a = [GtBox(10, 10, 50, 50)]
        gt_b = [GtBox(20, 20, 44, 44)]
        detector = make_fake_detector([Detection(10, 10, 50, 50, 0.95)])  # hits a, misses b

        result = evaluate_detector(
            detector,
            [(img_a, gt_a), (img_b, gt_b)],
            warmup=0,
            operating_threshold=0.5,
        )
        assert result.n_images == 2
        assert result.n_faces == 2
        assert result.recall == pytest.approx(0.5)
        assert result.precision == pytest.approx(0.5)  # 1 TP on a, 1 FP on b
        assert result.mean_fps > 0
        assert result.median_latency_ms >= 0

    def test_skips_unreadable_images(self, tmp_path: Path, make_fake_detector) -> None:
        bad = tmp_path / "bad.png"
        bad.write_bytes(b"not a png")
        result = evaluate_detector(make_fake_detector([]), [(bad, [])], warmup=0)
        assert result.n_images == 0


class TestRenderReport:
    def test_report_has_tables_and_caveat(self, tmp_path: Path, make_fake_detector) -> None:
        img = tmp_path / "a.png"
        _png(img)
        result = evaluate_detector(
            make_fake_detector([Detection(10, 10, 50, 50, 0.9)]),
            [(img, [GtBox(10, 10, 50, 50)])],
            warmup=0,
        )
        md = render_report([result], {"dataset": "x", "device": "cpu"})
        assert "# WIDER FACE" in md
        assert "not the official WIDER" in md
        assert "Recall by face size" in md
        assert result.detector in md
        assert md.count("|") > 20  # tables rendered
