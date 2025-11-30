import unittest
import cv2
import numpy as np
import os
from crop_balloons import crop_balloon_with_padding

class TestAdvancedCropping(unittest.TestCase):

    def setUp(self):
        """Set up a test image and output directory before each test."""
        # Create a dummy black image (300x400)
        self.test_image = np.zeros((300, 400, 3), dtype=np.uint8)
        self.output_dir = "test_output"
        self.output_path = os.path.join(self.output_dir, "test_crop.jpg")
        os.makedirs(self.output_dir, exist_ok=True)

    def tearDown(self):
        """Clean up created files and directory after each test."""
        if os.path.exists(self.output_path):
            os.remove(self.output_path)
        if os.path.exists(self.output_dir):
            os.rmdir(self.output_dir)

    def test_crop_center_with_padding(self):
        """
        Test cropping a bounding box in the center of the image with default padding.
        """
        # Bounding box in the center
        x1, y1, x2, y2 = 150, 100, 250, 200
        padding = 0.2  # 20%

        crop_balloon_with_padding(self.test_image, x1, y1, x2, y2, padding, self.output_path)

        # Verify the output
        self.assertTrue(os.path.exists(self.output_path))
        cropped_image = cv2.imread(self.output_path)
        self.assertIsNotNone(cropped_image)
        height, width, _ = cropped_image.shape
        self.assertEqual(height, width, "Cropped image should be a square.")

    def test_crop_near_edge(self):
        """
        Test that cropping a box near the edge of the image does not create
        an out-of-bounds crop and still produces a square.
        """
        # Bounding box near the top-left edge
        x1, y1, x2, y2 = 5, 5, 55, 55
        padding = 0.1  # 10%

        crop_balloon_with_padding(self.test_image, x1, y1, x2, y2, padding, self.output_path)

        # Verify the output
        self.assertTrue(os.path.exists(self.output_path))
        cropped_image = cv2.imread(self.output_path)
        self.assertIsNotNone(cropped_image)
        height, width, _ = cropped_image.shape
        self.assertEqual(height, width, "Cropped image near edge should be a square.")

if __name__ == '__main__':
    unittest.main()
