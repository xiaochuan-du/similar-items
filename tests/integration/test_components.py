import sagemaker
from sagemaker.processing import FrameworkProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput

import os
from pathlib import Path

BASE_DIR = Path(os.path.dirname(os.path.realpath(__file__)))

CODE_DIR = str(BASE_DIR.parent.parent / "sim_items_src")


def test_preprocessing(sys_config, sagemaker_session):
    instance_type = "ml.m5.xlarge"
    framework_version = "1.0-1"
    processing_instance_count = 1
    role = sys_config["role"]
    default_bucket = sys_config["default_bucket"]
    base_job_prefix = "test-preprocessing"
    input_data = f"s3://{default_bucket}/similiar-item/raw/"

    est_cls = sagemaker.sklearn.estimator.SKLearn

    pre_processor = FrameworkProcessor(
        framework_version=framework_version,
        instance_type=instance_type,
        instance_count=processing_instance_count,
        role=role,
        sagemaker_session=sagemaker_session,
        estimator_cls=est_cls,
        base_job_name=f"{base_job_prefix}Split",
    )

    processor_args = pre_processor.run(
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
        ],
        code="pre_processing.py",
        source_dir=CODE_DIR,
        wait=True,
    )
