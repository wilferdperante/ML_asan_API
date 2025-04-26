from flask import Flask
from flask_restx import Api, Resource, fields
import pandas as pd
import pickle

app = Flask(__name__)
api = Api(
    app,
    version='1.0',
    title='Ecommerce Customer API',
    description='Predict yearly spending based on customer behavior',
    doc='/docs'  # This enables Swagger UI at /docs
)

# Load the trained model
try:
    model = pickle.load(open("model.pkl", "rb"))
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Define the input model for Swagger documentation
input_model = api.model('PredictionInput', {
    'Avg. Session Length': fields.Float(required=True, example=34.49),
    'Time on App': fields.Float(required=True, example=12.65),
    'Time on Website': fields.Float(required=True, example=39.57),
    'Length of Membership': fields.Float(required=True, example=4.08)
})

# Define the response model
response_model = api.model('PredictionOutput', {
    'prediction': fields.List(fields.Float),
    'status': fields.String
})

@api.route('/predict')
class Predict(Resource):
    @api.expect(input_model)
    @api.marshal_with(response_model)
    def post(self):
        """Make a prediction for customer spending"""
        if model is None:
            api.abort(500, "Model not loaded")

        try:
            # Get JSON data from request
            json_data = api.payload
            
            # Convert to DataFrame
            input_data = pd.DataFrame([json_data])
            
            # Make prediction
            prediction = model.predict(input_data)
            
            return {
                "prediction": prediction.tolist(),
                "status": "success"
            }
        except Exception as e:
            api.abort(400, str(e))

@api.route('/health')
class Health(Resource):
    def get(self):
        """Service health check"""
        return {"status": "healthy"}

if __name__ == '__main__':
    app.run(debug=True)