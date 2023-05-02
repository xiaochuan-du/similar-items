# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""A CLI to create or update and run pipelines."""
from __future__ import absolute_import

import argparse
import json
import sys

from pipelines._utils import (
    get_pipeline_driver,
    convert_struct,
    get_pipeline_custom_tags,
)


def main():  # pragma: no cover
    """The main harness that creates or updates and runs the pipeline.

    Creates or updates the pipeline and runs it.
    """
    parser = argparse.ArgumentParser(
        "Creates or updates and runs the pipeline for the pipeline script."
    )

    parser.add_argument(
        "-n",
        "--module-name",
        dest="module_name",
        type=str,
        help="The module name of the pipeline to import.",
    )
    parser.add_argument(
        "-kwargs",
        "--kwargs",
        dest="kwargs",
        default=None,
        help="Dict string of keyword arguments for the pipeline generation (if supported)",
    )
    parser.add_argument(
        "-role-arn",
        "--role-arn",
        dest="role_arn",
        type=str,
        help="The role arn for the pipeline service execution role.",
    )
    parser.add_argument(
        "-description",
        "--description",
        dest="description",
        type=str,
        default=None,
        help="The description of the pipeline.",
    )
    parser.add_argument(
        "-tags",
        "--tags",
        dest="tags",
        default=None,
        help="""List of dict strings of '[{"Key": "string", "Value": "string"}, ..]'""",
    )
    args = parser.parse_args()

    if args.module_name is None or args.role_arn is None:
        parser.print_help()
        sys.exit(2)
    tags = convert_struct(args.tags)

    try:
        pipeline = get_pipeline_driver(args.module_name, args.kwargs)
        print(
            "###### Creating/updating a SageMaker Pipeline with the following definition:"
        )
        parsed = json.loads(pipeline.definition())
        print(json.dumps(parsed, indent=2, sort_keys=True))

        all_tags = get_pipeline_custom_tags(args.module_name, args.kwargs, tags)

        upsert_response = pipeline.upsert(
            role_arn=args.role_arn, description=args.description, tags=all_tags
        )
        print("\n###### Created/Updated SageMaker Pipeline: Response received:")
        print(upsert_response)

        # Parameters should be passed for the first execution of this pipeline so all
        # the baselines calculated will be registered as DriftCheckBaselines.
        # After the first execution, the default parameters can be used.

        execution = pipeline.start(
            parameters=dict(
                SkipDataQualityCheck=True,  # skip drift check for data quality
                RegisterNewDataQualityBaseline=True,  # register newly calculated baseline for data quality
                SkipDataBiasCheck=True,  # skip drift check for data bias
                RegisterNewDataBiasBaseline=True,  # register newly calculated baseline for data bias
                SkipModelQualityCheck=True,  # skip drift check for model quality
                RegisterNewModelQualityBaseline=True,  # register newly calculated baseline for model quality
                SkipModelBiasCheck=True,  # skip drift check for model bias
                RegisterNewModelBiasBaseline=True,  # register newly calculated baseline for model bias
                SkipModelExplainabilityCheck=True,  # skip drift check for model explainability
                RegisterNewModelExplainabilityBaseline=True,  # register newly calculated baseline for explainability
            )
        )

        # Update above code as below to use default parameter values for future pipeline executions
        # after approving the model registered by the first pipeline execution in Model Registry
        # so that all the checks are enabled and previous baselines are retained.

        # execution = pipeline.start()

        print(f"\n###### Execution started with PipelineExecutionArn: {execution.arn}")

        print("Waiting for the execution to finish...")
        execution.wait()
        print("\n#####Execution completed. Execution step details:")

        print(execution.list_steps())
        # Todo print the status?
    except Exception as e:  # pylint: disable=W0703
        print(f"Exception: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
