import os
import sys
import unittest
from fastapi.testclient import TestClient

# Ensure project root is on sys.path so `from FastAPI.app import app` works when
# running tests from the repository root.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from FastAPI.app import app


class FastAPIAppTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_home_page(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        # The root returns JSON like {"Hello":"World"}
        self.assertIn('Hello', response.json())

    def test_predict_page(self):
        payload = {"text": "I love this!"}
        response = self.client.post('/predict', json=payload)
        self.assertEqual(response.status_code, 200)
        # Returned value is a simple string 'Positive' or 'Negative'
        self.assertTrue(
            response.text.strip('"') in ("Positive", "Negative"),
            "Response should contain either 'Positive' or 'Negative'"
        )


if __name__ == '__main__':
    unittest.main()