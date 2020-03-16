from numpy.testing import assert_array_equal
import numpy as np
import PIL.Image

from testutil import testcase
from testutil import testsize
from thumbnails.transforms import RGBTransform

# testcase.TestCase is a KA-specific version of unittest.TestCase;
# the differences aren't really important to this test.
class RGBTransformTest(testcase.TestCase):
    """Test the behavior of thumbnails.transforms.RGBTransform.

    Most tests simply check a transformation's matrix representation
    against painstakingly hand-computed matrices for the same transformation.

    The test_rgb and test_rgba methods work with actual PIL images.
    """

    @testsize.tiny
    def test_desaturate(self):
        transform = RGBTransform().desaturate(factor=0.75,
                                              weights=[0.25, 0.50, 0.25])
        expected_matrix = [[0.4375, 0.375, 0.1875, 0],
                           [0.1875, 0.625, 0.1875, 0],
                           [0.1875, 0.375, 0.4375, 0]]
        assert_array_equal(transform.get_matrix(), np.array(expected_matrix))

    @testsize.tiny
    def test_multiply_with(self):
        transform = RGBTransform().multiply_with((255, 127.5, 0),
                                                 factor=0.75)
        expected_matrix = [[1.000, 0.000, 0.000, 0],
                           [0.000, 0.625, 0.000, 0],
                           [0.000, 0.000, 0.250, 0]]
        assert_array_equal(transform.get_matrix(), np.array(expected_matrix))

    @testsize.tiny
    def test_mix_with(self):
        transform = RGBTransform().mix_with((255, 127.5, 0),
                                            factor=0.5)
        expected_matrix = [[0.500, 0.000, 0.000, 127.5],
                           [0.000, 0.500, 0.000, 63.75],
                           [0.000, 0.000, 0.500, 0.000]]
        assert_array_equal(transform.get_matrix(), np.array(expected_matrix))

    @testsize.tiny
    def test_combination(self):
        # Test that chaining transforms works properly.
        # These particular transforms don't commute:
        # if the desaturation happened after the mixing,
        # the image would be...well, less saturated.
        transform = (RGBTransform()
                     .desaturate(factor=0.75, weights=[0.25, 0.50, 0.25])
                     .mix_with((0, 0, 255), factor=0.5))
        expected_matrix = [[0.21875, 0.1875, 0.09375, 0.000],
                           [0.09375, 0.3125, 0.09375, 0.000],
                           [0.09375, 0.1875, 0.21875, 127.5]]
        assert_array_equal(transform.get_matrix(), np.array(expected_matrix))

    @testsize.small
    def test_rgb(self):
        image = PIL.Image.new('RGB', (100, 50), (255, 0, 0))
        green_tint = RGBTransform().mix_with((0, 255, 0), factor=0.25)
        filtered_image = green_tint.applied_to(image)

        image_data = np.asarray(filtered_image)
        expected_color = np.asarray([191, 64, 0])  # rounded to nearest whole
        expected_image = np.apply_along_axis(lambda pixel: expected_color,
                                             len(image_data.shape) - 1,
                                             image_data)
        assert_array_equal(image_data, expected_image)

    @testsize.small
    def test_rgba(self):
        image = PIL.Image.new('RGBA', (100, 50), (255, 0, 0, 127))
        green_tint = RGBTransform().mix_with((0, 255, 0), factor=0.25)
        filtered_image = green_tint.applied_to(image)

        image_data = np.asarray(filtered_image)
        expected_color = np.asarray([191, 64, 0, 127])
        expected_image = np.apply_along_axis(lambda pixel: expected_color,
                                             len(image_data.shape) - 1,
                                             image_data)
        assert_array_equal(image_data, expected_image)

    @testsize.tiny
    def test_rgb_pixel(self):
        transform = (RGBTransform()
                     .desaturate(factor=0.5, weights=(0.25, 0.50, 0.25))
                     .multiply_with((255, 0, 255)))

        original = (255, 255, 0)
        # desaturated = (191.25, 191.25, 191.25)
        # partially_desaturated = (223.125, 223.125, 95.625)
        # multiplied = (223.125, 0, 95.625)
        rounded = (223, 0, 96)

        self.assertEqual(transform.applied_to_pixel(original), rounded)

    @testsize.tiny
    def test_rgba_pixel(self):
        transform = (RGBTransform()
                     .desaturate(factor=0.5, weights=(0.25, 0.50, 0.25))
                     .multiply_with((255, 0, 255)))

        original = (255, 255, 0, 122.8)
        # desaturated = (191.25, 191.25, 191.25, 122.8)
        # partially_desaturated = (223.125, 223.125, 95.625, 122.8)
        # multiplied = (223.125, 0, 95.625, 122.8)
        rounded = (223, 0, 96, 123)

        self.assertEqual(transform.applied_to_pixel(original), rounded)
