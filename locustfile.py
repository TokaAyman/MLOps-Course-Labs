from locust import HttpUser, task, between

class APIUser(HttpUser):
    wait_time = between(1, 3)  # wait between 1 and 3 seconds between tasks

    @task(2)
    def get_docs(self):
        self.client.get("/docs")

    @task(5)
    def predict(self):
        # Replace this JSON with the actual expected input shape for your predict API
        json_data = {
            "feature1": 0.5,
            "feature2": 1.5,
            "feature3": 3.0
        }
        self.client.post("/predict", json=json_data)
