"""Regression coverage for browser-decoded gzip responses from GitHub Pages."""

import gzip
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from companion_data import read_csv


class CsvLoadingTests(unittest.TestCase):
    def test_native_course_files_preserve_parser_options(self):
        root = Path(__file__).resolve().parents[1] / "public"
        for name, options in (
            ("height_data.csv", {}),
            ("rat_data.csv", {"comment": "#"}),
            ("peroxidase.csv", {}),
            ("Wasserqualitaet.csv", {"encoding": "utf-8-sig"}),
        ):
            with self.subTest(name=name):
                pd.testing.assert_frame_equal(
                    read_csv(root / name, **options),
                    pd.read_csv(root / name, **options),
                )

    def test_browser_decoded_gzip_response(self):
        payload = b"\xef\xbb\xbf# lecture data\ngender,height\nfemale,163.0\n"

        def response(*args, **kwargs):
            stream = BytesIO(payload)
            stream.headers = {"Content-Encoding": "gzip"}
            return stream

        url = "https://example.org/course/public/height_data.csv"
        options = {"encoding": "utf-8-sig", "comment": "#"}
        # Reproduce pandas attempting to decompress already-decoded bytes.
        with (
            patch("pandas.io.common.urlopen", side_effect=response),
            self.assertRaises(gzip.BadGzipFile),
        ):
            pd.read_csv(url, **options)

        with (
            patch("companion_data.sys.platform", "emscripten"),
            patch("companion_data.urlopen", side_effect=response) as fetch,
        ):
            result = read_csv(url, **options)
        fetch.assert_called_once_with(url)
        self.assertEqual(
            result.to_dict("records"), [{"gender": "female", "height": 163.0}]
        )


if __name__ == "__main__":
    unittest.main()
