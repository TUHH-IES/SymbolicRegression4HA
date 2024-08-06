
import unittest

from sr4ha.core.processed_data import SegmentedData
import polars as pl

class TestSegmentedData(unittest.TestCase):

    def test_get_segmentation_deviation(self):
        segmented_data = SegmentedData("tests/test_data/empty.csv", "tests/test_data/empty.csv", [0, 100, 200, 300, 400], "target_var")
        deviation = segmented_data.get_segmentation_deviation("tests/test_data/gt_switches.csv")
        self.assertEqual(deviation, 0)

    def test_get_segmentation_deviation_2(self):
        segmented_data = SegmentedData("tests/test_data/empty.csv", "tests/test_data/empty.csv", [0, 100, 300, 400], "target_var")
        deviation = segmented_data.get_segmentation_deviation("tests/test_data/gt_switches.csv")
        self.assertEqual(deviation, 100)

    def test_get_segmentation_deviation_3(self):
        segmented_data = SegmentedData("tests/test_data/empty.csv", "tests/test_data/empty.csv", [0, 100, 200, 300], "target_var")
        deviation = segmented_data.get_segmentation_deviation("tests/test_data/gt_switches.csv")
        self.assertEqual(deviation, 100)

    def test_get_segmentation_deviation_4(self):
        segmented_data = SegmentedData("tests/test_data/empty.csv", "tests/test_data/empty.csv", [0, 100, 150, 200, 300, 400], "target_var")
        deviation = segmented_data.get_segmentation_deviation("tests/test_data/gt_switches.csv")
        self.assertEqual(deviation, 100)

    def test_get_segmentation_deviation_5(self):
        segmented_data = SegmentedData("tests/test_data/empty.csv", "tests/test_data/empty.csv", [1, 99, 201, 301, 401], "target_var")
        deviation = segmented_data.get_segmentation_deviation("tests/test_data/gt_switches.csv")
        self.assertEqual(deviation, 1)

    def test_segmented_data_from_file(self):
        data = pl.read_csv("tests/test_data/empty.csv")
        segmented_data = SegmentedData.from_file(data, "target_var", "tests/test_data/segmentation_results.csv")
        self.assertEqual(segmented_data.switches, [0, 1000, 2000])
        self.assertEqual(segmented_data.target_var, "target_var")