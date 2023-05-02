"""Example workflow pipeline script for similar items
Implements a get_pipeline(**kwargs) method.
"""
import os
from pathlib import Path

import boto3
import sagemaker
import sagemaker.session
from sagemaker import PipelineModel, image_uris
from sagemaker.inputs import TrainingInput, TransformInput
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import FrameworkProcessor
from sagemaker.transformer import Transformer
from sagemaker.tuner import (
    CategoricalParameter,
    ContinuousParameter,
    HyperparameterTuner,
)
from sagemaker.workflow.steps import TrainingStep
from sagemaker.huggingface.model import HuggingFaceModel
from sagemaker.huggingface import HuggingFace
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.parameters import ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.steps import ProcessingStep, TransformStep, TuningStep
from sagemaker.workflow.steps import CacheConfig
from sagemaker.network import NetworkConfig

from .config import subnets, security_group

BASE_DIR = Path(os.path.dirname(os.path.realpath(__file__)))
CODE_DIR = str(BASE_DIR.parent.parent / "sim_items_src")
DEBUG_DIR = str(BASE_DIR.parent.parent / "debug_src")


def get_sagemaker_client(region):
    """Gets the sagemaker client.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")
    return sagemaker_client


def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )


def get_pipeline_custom_tags(new_tags, region, sagemaker_project_arn=None):
    try:
        sm_client = get_sagemaker_client(region)
        response = sm_client.list_tags(ResourceArn=sagemaker_project_arn)
        project_tags = response["Tags"]
        for project_tag in project_tags:
            new_tags.append(project_tag)
    except Exception as e:
        print(f"Error getting project tags: {e}")
    return new_tags


def get_pipeline(
    region,
    sagemaker_project_arn=None,
    role=None,
    default_bucket=None,
    default_input_s3_url="",
    processing_instance_count=1,
    transform_instance_type="ml.m5.xlarge",
    process_instance_type="ml.m5.xlarge",
    model_package_group_name="SimItemsPackageGroup",
    pipeline_name="SimItemsPipeline",
    base_job_prefix="SimItems",
):
    """Gets a SageMaker ML Pipeline instance working with on abalone data.

    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        an instance of a pipeline
    """
    sagemaker_session = get_session(region, default_bucket)
    pipeline_session = PipelineSession(
        sagemaker_client=sagemaker_session.sagemaker_client,
        boto_session=sagemaker_session.boto_session,
    )
    default_bucket = sagemaker_session.default_bucket()
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)

    # parameters for pipeline execution

    framework_version = "1.0-1"
    network_config = NetworkConfig(security_group_ids=[security_group], subnets=subnets)
    cache_config = CacheConfig(enable_caching=False, expire_after="PT1H")

    input_data = ParameterString(
        name="InputData",
        default_value=default_input_s3_url,
    )
    opensearch_url = ParameterString(
        name="OpenSearchUrl",
    )
    opensearch_index_name = ParameterString(
        name="OpenSearchIndexName",
    )
    opensearch_embeddings_dimension = ParameterString(
        name="OpensearchEmbeddingsDimension", default_value="768"
    )
    opensearch_ingestion_bz = ParameterString(
        name="OpensearchIngestionBz", default_value="1"
    )
    opensearch_num_threads = ParameterString(
        name="OpensearchNumThreads", default_value="1"
    )

    # status of newly trained model in registry
    model_approval_status = ParameterString(
        name="ModelApprovalStatus", default_value="Approved"
    )

    est_cls = sagemaker.sklearn.estimator.SKLearn

    sklearn_processor = FrameworkProcessor(
        framework_version=framework_version,
        instance_type=process_instance_type,
        instance_count=processing_instance_count,
        role=role,
        sagemaker_session=pipeline_session,
        estimator_cls=est_cls,
        base_job_name=f"{base_job_prefix}Split",
        network_config=network_config,
    )

    processor_args = sklearn_processor.run(
        inputs=[
            ProcessingInput(source=input_data, destination="/opt/ml/processing/input"),
        ],
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
            ProcessingOutput(
                output_name="validation", source="/opt/ml/processing/validation"
            ),
            ProcessingOutput(
                output_name="raw_test", source="/opt/ml/processing/test_raw"
            ),
            ProcessingOutput(
                output_name="raw_train", source="/opt/ml/processing/train_raw"
            ),
            ProcessingOutput(
                output_name="scaler_model", source="/opt/ml/processing/prep_model"
            ),
        ],
        code="pre_processing.py",
        source_dir=CODE_DIR,
    )

    step_process = ProcessingStep(
        name="SimItemsProcess", step_args=processor_args, cache_config=cache_config
    )

    hyperparameters = {
        "model_name_or_path": "distilbert-base-uncased",
        "output_dir": "/opt/ml/model",
        "train_file": "/opt/ml/input/data/train/train.csv",
        "validation_file": "/opt/ml/input/data/validation/validation.csv",
        "do_train": True,
        "do_eval": True,
        # TODO
        # "num_train_epochs": 5,
        "num_train_epochs": 1,
        "save_total_limit": 1,
        # add your remaining hyperparameters
        # more info here https://github.com/huggingface/transformers/tree/v4.10.0/examples/pytorch/text-classification
    }

    # git configuration to download our fine-tuning script
    git_config = {
        "repo": "https://github.com/huggingface/transformers.git",
        "branch": "v4.17.0",
    }

    # creates Hugging Face estimator
    huggingface_estimator_bert = HuggingFace(
        entry_point="run_glue.py",  # note we are pointing to the processing script in HF repo
        source_dir="./examples/pytorch/text-classification",
        instance_type="ml.p3.2xlarge",
        instance_count=1,
        role=role,
        git_config=git_config,
        transformers_version="4.17.0",
        pytorch_version="1.10.2",
        py_version="py38",
        hyperparameters=hyperparameters,
        disable_profiler=True,
        sagemaker_session=pipeline_session,
    )
    train_args = huggingface_estimator_bert.fit(
        {
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "train"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "validation"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
        }
    )
    step_train = TrainingStep(
        name="SimItemTraining", step_args=train_args, cache_config=cache_config
    )

    # create Hugging Face Model Class
    huggingface_model = sagemaker.huggingface.HuggingFaceModel(
        env={"HF_TASK": "text-classification", "HF_EMB": "true"},
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        role=role,
        transformers_version="4.17.0",
        pytorch_version="1.10.2",
        py_version="py38",
        entry_point="inference.py",
        source_dir=CODE_DIR,
        sagemaker_session=pipeline_session,
    )

    scaler_model_s3 = "{}/model.tar.gz".format(
        step_process.arguments["ProcessingOutputConfig"]["Outputs"][-1]["S3Output"][
            "S3Uri"
        ]
    )
    scaler_model = SKLearnModel(
        model_data=scaler_model_s3,
        role=role,
        sagemaker_session=pipeline_session,
        entry_point="pre_processing.py",
        source_dir=CODE_DIR,
        framework_version=framework_version,
    )

    # TODO how do we cache transform step
    pipeline_model = PipelineModel(
        models=[scaler_model, huggingface_model],
        role=role,
        sagemaker_session=pipeline_session,
    )

    step_create_model = ModelStep(
        name="SimItemsCreateModel",
        step_args=pipeline_model.create(
            instance_type=transform_instance_type,
        ),
    )

    transformer = Transformer(
        model_name=step_create_model.properties.ModelName,
        instance_type=transform_instance_type,
        # TODO
        instance_count=5,
        output_path=f"s3://{default_bucket}/test/SimItems/",
        max_payload=1,
        accept="application/json",
        assemble_with="Line",
        sagemaker_session=pipeline_session,
    )

    step_transform = TransformStep(
        name="SimItemsTransform",
        transformer=transformer,
        inputs=TransformInput(
            data=step_process.properties.ProcessingOutputConfig.Outputs[
                "raw_train"
            ].S3Output.S3Uri,
            content_type="text/csv",
            split_type="Line",
        ),
        cache_config=cache_config,
    )
    est_cls = sagemaker.sklearn.estimator.SKLearn

    knn_indexing_processor = FrameworkProcessor(
        estimator_cls=est_cls,
        framework_version=framework_version,
        instance_type=process_instance_type,
        instance_count=processing_instance_count,
        role=role,
        sagemaker_session=pipeline_session,
        base_job_name=f"{base_job_prefix}Indexing",
        network_config=network_config,
    )

    processor_args = knn_indexing_processor.run(
        inputs=[
            ProcessingInput(
                source=step_process.properties.ProcessingOutputConfig.Outputs[
                    "raw_train"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/input_train",
            ),
            ProcessingInput(
                source=step_transform.properties.TransformOutput.S3OutputPath,
                destination="/opt/ml/processing/embeddings",
            ),
        ],
        code="knn_index.py",
        source_dir=CODE_DIR,
        arguments=[
            "--opensearch_url",
            opensearch_url,
            "--opensearch_region",
            region,
            "--opensearch_index_name",
            opensearch_index_name,
            "--opensearch_embeddings_dimension",
            opensearch_embeddings_dimension,
            "--opensearch_ingestion_bz",
            opensearch_ingestion_bz,
            "--opensearch_num_threads",
            opensearch_num_threads,
        ],
    )
    step_indexing = ProcessingStep(
        name="SimItemsIndexing", step_args=processor_args, cache_config=cache_config
    )

    # create Hugging Face Model Class
    huggingface_model_knn = sagemaker.huggingface.HuggingFaceModel(
        env={
            "HF_TASK": "text-classification",
            "HF_EMB": "true",
            "ES_HOST": opensearch_url,
            "ES_REGION": region,
            "ES_K": "5",
            "ES_INDEX_NAME": "sim-item",
            "HF_KNN": "true",
        },
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        role=role,
        transformers_version="4.17.0",
        pytorch_version="1.10.2",
        py_version="py38",
        entry_point="inference.py",
        source_dir=CODE_DIR,
        sagemaker_session=pipeline_session,
    )
    pipeline_model_inference = PipelineModel(
        models=[scaler_model, huggingface_model_knn],
        role=role,
        sagemaker_session=pipeline_session,
        vpc_config={"Subnets": subnets, "SecurityGroupIds": [security_group]},
    )

    register_args = pipeline_model_inference.register(
        content_types=["text/csv"],
        response_types=["application/json"],
        inference_instances=["ml.m5.large", "ml.m5.xlarge"],
        transform_instances=["ml.m5.xlarge"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
    )

    step_register_pipeline_model = ModelStep(
        name="PipelineModel", step_args=register_args, depends_on=[step_indexing]
    )

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            input_data,
            model_approval_status,
            opensearch_url,
            opensearch_index_name,
            opensearch_embeddings_dimension,
            opensearch_ingestion_bz,
            opensearch_num_threads,
        ],
        steps=[
            step_process,
            step_train,
            step_register_pipeline_model,
            step_create_model,
            step_transform,
            step_indexing,
        ],
        sagemaker_session=pipeline_session,
    )
    return pipeline
