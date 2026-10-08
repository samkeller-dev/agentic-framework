import os

from pydantic_ai import models

models.ALLOW_MODEL_REQUESTS = False
os.environ["PYDANTIC_AI_NO_BANNER"] = "1"
