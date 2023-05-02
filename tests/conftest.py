import pytest
from botocore.exceptions import ClientError
import os
import boto3
import sagemaker
import logging
from sagemaker import ModelPackage
import time
import tarfile
import zipfile
from configparser import ConfigParser
from pipelines.sim_items.pipeline import get_session


logger = logging.getLogger(__name__)
sm_client = boto3.client("sagemaker")


@pytest.fixture
def sys_config():

    config_ = ConfigParser()
    config_.read("tox.ini")
    return config_["sim-items"]


@pytest.fixture
def sagemaker_session(sys_config):
    default_bucket = sys_config["default_bucket"]
    region = sys_config["region"]
    return get_session(region, default_bucket)


def get_approved_package(model_package_group_name):
    """Gets the latest approved model package for a model package group.

    Args:
        model_package_group_name: The model package group name.

    Returns:
        The SageMaker Model Package ARN.
    """
    try:
        # Get the latest approved model package
        response = sm_client.list_model_packages(
            ModelPackageGroupName=model_package_group_name,
            ModelApprovalStatus="Approved",
            SortBy="CreationTime",
            MaxResults=100,
        )
        approved_packages = response["ModelPackageSummaryList"]

        # Fetch more packages if none returned with continuation token
        while len(approved_packages) == 0 and "NextToken" in response:
            logger.debug(
                "Getting more packages for token: {}".format(response["NextToken"])
            )
            response = sm_client.list_model_packages(
                ModelPackageGroupName=model_package_group_name,
                ModelApprovalStatus="Approved",
                SortBy="CreationTime",
                MaxResults=100,
                NextToken=response["NextToken"],
            )
            approved_packages.extend(response["ModelPackageSummaryList"])

        # Return error if no packages found
        if len(approved_packages) == 0:
            error_message = f"No approved ModelPackage found for ModelPackageGroup: {model_package_group_name}"
            logger.error(error_message)
            raise Exception(error_message)

        # Return the pmodel package arn
        model_package_arn = approved_packages[0]["ModelPackageArn"]
        logger.info(
            f"Identified the latest approved model package: {model_package_arn}"
        )
        return approved_packages[0]
        # return model_package_arn
    except ClientError as e:
        error_message = e.response["Error"]["Message"]
        logger.error(error_message)
        raise Exception(error_message)


@pytest.fixture
def inference_input():
    input_ = """Paper,B000CSB74S,"INKPRESS BACKLIGHT FILM- 7 Mil, 11"" x 17""- 20 sheetsThickness:7 milBrightness:N/AWeight:N/AInkpress Backlight film isideal for sophisticated graphics. A translucent polyester flim that ismatte on one side and glossy on the other, the films ultra whitediffusing layer allows reverse or direct print for either finish.Compatible with both dye and pigment ink printers, the matte side isdesigned for mirror-type printing for use with illuminated frames andreverse lighting setups.Sheets: 8.5x11; 11x17; 13x19; 17x22Rolls: 17""x100'; 24""x100'; 44""x100'",41.11,http://ecx.images-amazon.com/images/I/41GPSkFWtOL._SY300_.jpg,{'also_viewed': ['B0010CFZUM']},,"[['Office Products', 'Office & School Supplies', 'Paper', 'Inkjet Printer Paper']]","Inkpress Backlight Film- 7 Mil, 11&quot; x 17&quot;- 20 sheets",\n"""
    return input_


@pytest.fixture
def inference_endpoint_name(scope="module"):
    model_package_group_name = "SimItemsModelPackageGroup"
    endpoint_name = "SIM-ITEMS-test-endpoint"
    role = "arn:aws:iam::071965382733:role/clip-poc-stack-SageMakerExecutionRole-GDUNQP4Z65X9"

    sm_client = boto3.client("sagemaker")

    pck = get_approved_package(
        model_package_group_name
    )  # Reminder: model_package_group_name was defined as "NominetAbaloneModelPackageGroupName" at the beginning of the pipeline definition
    model_description = sm_client.describe_model_package(
        ModelPackageName=pck["ModelPackageArn"]
    )

    sess = boto3.Session()

    sagemaker_session = sagemaker.Session(boto_session=sess)

    model_package_arn = model_description["ModelPackageArn"]
    model = ModelPackage(
        role=role,
        model_package_arn=model_package_arn,
        sagemaker_session=sagemaker_session,
    )
    try:
        _ = sm_client.describe_endpoint(EndpointName=endpoint_name)
    except ClientError:
        model.deploy(
            initial_instance_count=1,
            instance_type="ml.m5.xlarge",
            endpoint_name=endpoint_name,
        )
    yield endpoint_name
    sm_client.delete_endpoint(EndpointName=endpoint_name)
    sm_client.delete_endpoint_config(EndpointConfigName=endpoint_name)
