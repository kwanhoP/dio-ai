import json
from pathlib import Path
from typing import Any, Dict

import boto3
import botocore.config

DEFAULT_CONFIG = {"max_pool_connections": 50}
RESOURCE_DIR = Path(__file__).resolve().parent
with open(RESOURCE_DIR.joinpath("s3.json"), "r") as f_in:
    BOTO3_CONFIG = json.load(f_in)


def get_s3_client(boto3_config: Dict[str, Any] = BOTO3_CONFIG):
    client_config = botocore.config.Config(**DEFAULT_CONFIG)
    return boto3.client(**boto3_config, config=client_config)
