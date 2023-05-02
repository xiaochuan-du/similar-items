#!/usr/bin/env python

from configparser import ConfigParser
import boto3
import sagemaker


if __name__ == "__main__":

    cfn_stack_name = "sm-os-stack"
    tox_file_path = "tox.ini"
    code_config_path = "./pipelines/sim_items/config.py"
    sagemaker_session = sagemaker.Session()
    bucket = (
        sagemaker_session.default_bucket()
    )  # replace with your own bucket if you have one
    s3 = sagemaker_session.boto_session.resource("s3")

    raw_data_url = f"s3://{bucket}/similiar-item/raw/data.csv"

    region = boto3.Session().region_name
    role = sagemaker.get_execution_role()

    config_ = ConfigParser()
    config_.read(tox_file_path)

    client = boto3.client("cloudformation")

    response = client.describe_stacks(
        StackName=cfn_stack_name,
    )

    my_session = boto3.session.Session()

    opensearch_region = my_session.region_name

    def get_cfn_stack_output(key):
        return [
            output_["OutputValue"]
            for output_ in response["Stacks"][0]["Outputs"]
            if output_["OutputKey"] == key
        ][0]

    opensearch_url = get_cfn_stack_output("DomainEndpoint")

    # add pipeline VPC config
    security_group = get_cfn_stack_output("SecurityGroup")

    subnets = [
        get_cfn_stack_output("VpcPrivateSubnet1"),
        get_cfn_stack_output("VpcPrivateSubnet2"),
    ]
    code_config_content = (
        f"""\nsubnets = {subnets}\nsecurity_group = "{security_group}" """
    )
    with open(code_config_path, "a") as file_object:
        file_object.write(code_config_content)

    # add test config
    if "sim-items:test" not in set(config_.keys()):
        config_["sim-items:test"] = dict(
            role=role,
            default_bucket=bucket,
            region=region,
            test_pipeline_prefix="TestMlops",
            test_model_package_group="TestSimItemsModelPackageGroup",
            test_raw_input_s3_url=raw_data_url,
            opensearch_url=opensearch_url,
            opensearch_index_name="test-sim-item",
        )

    # add notebook config
    if "sim-items:notebook" not in set(config_.keys()):
        config_["sim-items:notebook"] = dict(
            role=role,
            default_bucket=bucket,
            region=region,
            pipeline_name="SimItemsPipeline-Example",
            model_package_group_name="SimItemsModelPackageGroup-Example",
            raw_data_url=raw_data_url,
            opensearch_url=opensearch_url,
            opensearch_index_name="sim-item",
            endpoint_name="SIM-ITEMS-test-endpoint",
            security_group=security_group,
            subnets=subnets,
        )

    with open(tox_file_path, "w") as configfile:
        config_.write(configfile)

    print(
        f"Config setup completed! The config file can be found in {tox_file_path} and {code_config_path}"
    )
