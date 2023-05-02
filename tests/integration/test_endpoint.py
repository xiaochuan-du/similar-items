from sagemaker.predictor import Predictor
from sagemaker.deserializers import JSONDeserializer
from sagemaker.serializers import CSVSerializer


def test_endpoint(inference_input, inference_endpoint_name):
    predictor = Predictor(
        endpoint_name=inference_endpoint_name,
        serializer=CSVSerializer(),
        deserializer=JSONDeserializer(),
        content_type="text/csv",
        accept="application/json",
    )
    res = predictor.predict(inference_input)
    assert len(res) > 0
