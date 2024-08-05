
import unittest

from sr4ha.core.processed_data import SegmentedData

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