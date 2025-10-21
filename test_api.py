# Create test script
import requests
import numpy as np

# Generate sample signal
signal = np.random.rand(900, 3).tolist()

# Send prediction request
response = requests.post(
    'http://localhost:8080/predict',
    json={'signal': signal}
)

print("Response:", response.json())
