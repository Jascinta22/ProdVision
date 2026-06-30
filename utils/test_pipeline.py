"""
test_pipeline.py
Unit tests validating the regex/parsing helper functions in parser.py and
the preprocessing helpers in preprocess.py. Run with:

    python -m pytest utils/test_pipeline.py -v
"""

import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parser import (
    _extract_weight, _extract_mrp, _extract_dates, parse_attributes,
    compute_confidence_score,
)
from preprocess import resize_image, to_grayscale, denoise, threshold


class TestParserRegex(unittest.TestCase):

    def test_extract_weight_grams(self):
        self.assertEqual(_extract_weight("Net Wt: 250g pack"), "250 g")

    def test_extract_weight_ml(self):
        self.assertEqual(_extract_weight("Volume 500 ml bottle"), "500 ml")

    def test_extract_weight_kg(self):
        self.assertEqual(_extract_weight("Pack size 2kg"), "2 kg")

    def test_extract_weight_none(self):
        self.assertIsNone(_extract_weight("No weight info here"))

    def test_extract_mrp_with_symbol(self):
        self.assertEqual(_extract_mrp("MRP ₹120.00 incl. of taxes"), "120.00")

    def test_extract_mrp_with_rs(self):
        self.assertEqual(_extract_mrp("Price Rs. 45"), "45")

    def test_extract_dates_labeled(self):
        text = "MFG 01/2024 EXP 01/2026"
        mfg, exp = _extract_dates(text)
        self.assertEqual(mfg, "01/2024")
        self.assertEqual(exp, "01/2026")

    def test_extract_dates_fallback(self):
        text = "Some text 05/03/2024 then 05/03/2026 randomly placed"
        mfg, exp = _extract_dates(text)
        self.assertIsNotNone(mfg)
        self.assertIsNotNone(exp)

    def test_extract_best_before_months(self):
        text = "MFG 01/2025 best before 12 months"
        mfg, exp = _extract_dates(text)
        self.assertEqual(mfg, "01/2025")
        self.assertIsNotNone(exp)


class TestFullParse(unittest.TestCase):

    def test_parse_attributes_amul_sample(self):
        sample_text = (
            "AMUL TAAZA\nToned Milk\nNet Vol: 500 ml\n"
            "MRP Rs. 28.00\nMFG 01/2026\nEXP 10/2026"
        )
        attrs = parse_attributes(sample_text)
        self.assertEqual(attrs["brand"], "Amul")
        self.assertEqual(attrs["weight"], "500 ml")
        self.assertEqual(attrs["mrp"], "28.00")

    def test_product_name_avoids_brand_for_amul_butter(self):
        sample_text = (
            "AMUL\n"
            "PASTEURISED\n"
            "utterly butterly delicious\n"
            "School Pack"
        )
        attrs = parse_attributes(sample_text)
        self.assertNotEqual(attrs["product_name"], "Amul")
        self.assertIn("butter", attrs["product_name"].lower())

    def test_compute_confidence_score_range(self):
        attrs = {"brand": "Amul", "product_name": "Milk", "weight": "500 ml",
                  "mrp": "28", "mfg_date": "01/2026", "exp_date": "10/2026"}
        score = compute_confidence_score(90.0, attrs)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)

    def test_compute_confidence_score_partial(self):
        attrs = {"brand": None, "product_name": None, "weight": None,
                  "mrp": None, "mfg_date": None, "exp_date": None}
        score = compute_confidence_score(80.0, attrs)
        self.assertEqual(score, round(0.6 * 80.0, 2))


class TestPreprocessHelpers(unittest.TestCase):

    def setUp(self):
        # Fake 100x200 BGR image (random noise)
        self.image = (np.random.rand(100, 200, 3) * 255).astype(np.uint8)

    def test_resize_image_width(self):
        resized = resize_image(self.image, width=400)
        self.assertEqual(resized.shape[1], 400)

    def test_to_grayscale_shape(self):
        gray = to_grayscale(self.image)
        self.assertEqual(len(gray.shape), 2)

    def test_denoise_preserves_shape(self):
        gray = to_grayscale(self.image)
        denoised = denoise(gray)
        self.assertEqual(denoised.shape, gray.shape)

    def test_threshold_binary_output(self):
        gray = to_grayscale(self.image)
        denoised = denoise(gray)
        binarized = threshold(denoised)
        unique_vals = set(np.unique(binarized).tolist())
        self.assertTrue(unique_vals.issubset({0, 255}))


if __name__ == "__main__":
    unittest.main()
