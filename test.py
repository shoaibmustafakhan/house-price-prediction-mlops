import unittest
import app
import json

class TestAPI(unittest.TestCase):

    def setUp(self):
        app.app.testing = True
        self.client = app.app.test_client()

    def test_predict(self):
        # Sample input data
        data = {
            "OverallQual": 7,
            "GrLivArea": 1710,
            "GarageCars": 2
            # Add other necessary features with default or median values
        }
        response = self.client.post('/predict', data=json.dumps(data), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.get_data(as_text=True))
        self.assertIn('prediction', data)
        self.assertIsInstance(data['prediction'], float)

if __name__ == '__main__':
    unittest.main()
