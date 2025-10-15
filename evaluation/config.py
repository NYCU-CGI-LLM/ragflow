#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import json
from pathlib import Path
from typing import Any, Dict, Optional

from ruamel.yaml import YAML

yaml = YAML(typ='safe')


class EvaluationConfig:
    """Configuration loader for RAGFlow evaluation.

    Exactly one of the following sections must be provided:
      - retrieval: Evaluate retrieval quality using a dataset (dataset_id/name)
      - generation: Evaluate retrieval+generation quality using a chat (chat_id/name)
    """

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.mode = self._determine_mode()
        self._validate_config()

    # -------------------------------------------------------------------------
    # Loading & validation
    # -------------------------------------------------------------------------
    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        file_ext = self.config_path.suffix.lower()
        with open(self.config_path, "r", encoding="utf-8") as fp:
            if file_ext in [".yaml", ".yml"]:
                try:
                    return yaml.load(fp)
                except Exception as exc:
                    raise ValueError(f"Failed to parse YAML config file: {exc}") from exc

            try:
                return json.load(fp)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse JSON config file: {exc}") from exc

    def _determine_mode(self) -> str:
        has_retrieval = "retrieval" in self.config
        has_generation = "generation" in self.config

        if has_retrieval and has_generation:
            raise ValueError(
                "Configuration must include only one of 'retrieval' or 'generation' sections."
            )
        if not has_retrieval and not has_generation:
            raise ValueError("Configuration must include a 'retrieval' or 'generation' section.")

        return "retrieval" if has_retrieval else "generation"

    def _validate_config(self):
        for section in ("dataset", "output"):
            if section not in self.config:
                raise ValueError(f"Missing required config section: {section}")

        if self.mode == "retrieval":
            retrieval = self.config["retrieval"]
            if "api_key" not in retrieval:
                raise ValueError("Missing 'api_key' in retrieval config.")
            if "dataset_id" not in retrieval and "dataset_name" not in retrieval:
                raise ValueError("Retrieval config requires either 'dataset_id' or 'dataset_name'.")
            if "chat_id" in retrieval or "chat_name" in retrieval:
                raise ValueError("'chat_id' or 'chat_name' should not appear in the retrieval config.")

        else:  # generation mode
            generation = self.config["generation"]
            if "api_key" not in generation:
                raise ValueError("Missing 'api_key' in generation config.")
            if "chat_id" not in generation and "chat_name" not in generation:
                raise ValueError("Generation config requires either 'chat_id' or 'chat_name'.")
            prohibited = {"vector_similarity_weight"}
            illegal_keys = prohibited.intersection(generation.keys())
            if illegal_keys:
                raise ValueError(
                    f"Generation config cannot define {', '.join(sorted(illegal_keys))}; "
                    "these parameters are controlled by the chat configuration."
                )
            if "size" in generation:
                size_value = generation["size"]
                if not isinstance(size_value, int) or size_value <= 0:
                    raise ValueError("Generation config 'size' must be a positive integer.")

        dataset_cfg = self.config["dataset"]
        if "path" not in dataset_cfg:
            raise ValueError("Missing 'path' in dataset config.")

        output_cfg = self.config["output"]
        if "results_path" not in output_cfg:
            raise ValueError("Missing 'results_path' in output config.")

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _active_section(self) -> Dict[str, Any]:
        return self.config[self.mode]

    @property
    def evaluation_mode(self) -> str:
        return self.mode

    # -------------------------------------------------------------------------
    # Shared properties
    # -------------------------------------------------------------------------
    @property
    def api_key(self) -> str:
        return self._active_section()["api_key"]

    @property
    def base_url(self) -> str:
        return self._active_section().get("base_url", "http://localhost:9380")

    @property
    def dataset_path(self) -> Path:
        return Path(self.config["dataset"]["path"])

    @property
    def dataset_limit(self) -> Optional[int]:
        return self.config["dataset"].get("limit")

    @property
    def results_path(self) -> Path:
        return Path(self.config["output"]["results_path"])

    @property
    def metrics(self) -> Optional[list]:
        section = self._active_section()
        metrics = section.get("metrics")
        if metrics is not None:
            return metrics
        return self.config.get("metrics")

    @property
    def k_values(self) -> Optional[list]:
        section = self._active_section()
        k_vals = section.get("k_values")
        if k_vals is not None:
            return k_vals
        return self.config.get("k_values")

    @property
    def dynamic_rerank_limit(self) -> bool:
        return self._active_section().get("dynamic_rerank_limit", True)

    # -------------------------------------------------------------------------
    # Retrieval-only properties
    # -------------------------------------------------------------------------
    @property
    def dataset_id(self) -> Optional[str]:
        if self.mode != "retrieval":
            return None
        return self.config["retrieval"].get("dataset_id")

    @property
    def dataset_name(self) -> Optional[str]:
        if self.mode != "retrieval":
            return None
        return self.config["retrieval"].get("dataset_name")

    @property
    def size(self) -> int:
        if self.mode != "retrieval":
            return 5
        return self.config["retrieval"].get("size", 5)

    @property
    def vector_similarity_weight(self) -> float:
        if self.mode != "retrieval":
            return 1.0
        return self.config["retrieval"].get("vector_similarity_weight", 1.0)

    @property
    def similarity_threshold(self) -> float:
        if self.mode != "retrieval":
            return self.config.get("generation", {}).get("similarity_threshold", 0.2)
        return self.config["retrieval"].get("similarity_threshold", 0.2)

    @property
    def top_k(self) -> int:
        if self.mode != "retrieval":
            return self.config.get("generation", {}).get("top_k", 1024)
        return self.config["retrieval"].get("k", 5)

    # -------------------------------------------------------------------------
    # Generation-only properties
    # -------------------------------------------------------------------------
    @property
    def chat_id(self) -> Optional[str]:
        if self.mode != "generation":
            return None
        return self.config["generation"].get("chat_id")

    @property
    def chat_name(self) -> Optional[str]:
        if self.mode != "generation":
            return None
        return self.config["generation"].get("chat_name")

    @property
    def generation_size(self) -> Optional[int]:
        if self.mode != "generation":
            return None
        return self.config["generation"].get("size")

    # -------------------------------------------------------------------------
    # Flags
    # -------------------------------------------------------------------------
    @property
    def generation_enabled(self) -> bool:
        return self.mode == "generation"
