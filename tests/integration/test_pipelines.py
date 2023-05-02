from pipelines.sim_items.pipeline import get_pipeline


def test_pipelines_execution(
    sys_config,
):

    region = sys_config["region"]
    role = sys_config["role"]
    default_bucket = sys_config["default_bucket"]
    test_pipeline_prefix = sys_config["test_pipeline_prefix"]
    test_model_package_group = sys_config["test_model_package_group"]
    test_raw_input_s3_url = sys_config["test_raw_input_s3_url"]
    opensearch_url = sys_config["opensearch_url"]
    opensearch_index_name = sys_config["opensearch_index_name"]

    # Change these to reflect your project/business name or if you want to separate ModelPackageGroup/Pipeline from the rest of your team
    pipeline_name = f"{test_pipeline_prefix}Index"
    pipeline = get_pipeline(
        region=region,
        role=role,
        default_bucket=default_bucket,
        model_package_group_name=test_model_package_group,
        pipeline_name=pipeline_name,
    )
    pipeline.upsert(role_arn=role)
    execution = pipeline.start(
        parameters=dict(
            InputData=test_raw_input_s3_url,
            OpenSearchUrl=opensearch_url,
            OpenSearchIndexName=opensearch_index_name,
        )
    )
    execution.wait()
