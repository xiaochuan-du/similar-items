import boto3
from requests_aws4auth import AWS4Auth
from elasticsearch import Elasticsearch, RequestsHttpConnection


def generate_features(df):
    df = df.assign(
        text="Title::" + df.title + "\n\n\n" + "Description::" + df.description
    )
    return df[["text"]]


def get_es_client(
    host,
    region,
    port=443,
):

    credentials = boto3.Session().get_credentials()
    awsauth = AWS4Auth(
        credentials.access_key,
        credentials.secret_key,
        region,
        "es",
        session_token=credentials.token,
    )

    es = Elasticsearch(
        hosts=[{"host": host, "port": port}],
        http_auth=awsauth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=60,  # for connection timeout errors
    )
    return es
