import unittest
import app
import json


class TestAPI(unittest.TestCase):

    def setUp(self):
        app.app.testing = True
        self.client = app.app.test_client()

    def test_predict(self):
        # Sample valid input data
        data = {
            "OverallQual": 7,
            "GrLivArea": 1710,
            "GarageCars": 2
        }
        response = self.client.post('/predict', data=json.dumps(data), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.get_data(as_text=True))
        self.assertIn('prediction', data)
        self.assertIsInstance(data['prediction'], float)

    def test_predict_invalid_input(self):
        # Test with missing fields
        data = {
            "OverallQual": 7  # Missing other necessary fields
        }
        response = self.client.post('/predict', data=json.dumps(data), content_type='application/json')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.get_data(as_text=True))
        self.assertIn('error', data)

    def test_predict_non_numeric_input(self):
        # Test with non-numeric input
        data = {
            "OverallQual": "abc",  # Invalid value
            "GrLivArea": 1710,
            "GarageCars": 2
        }
        response = self.client.post('/predict', data=json.dumps(data), content_type='application/json')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.get_data(as_text=True))
        self.assertIn('error', data)


if __name__ == '__main__':
    unittest.main()
