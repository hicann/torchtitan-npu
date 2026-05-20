# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Trim verbose model-structure blocks from smoke-test logs.

Converter "Applied ..." / "replacement(s)" status lines are intentionally kept.
"""

from __future__ import annotations

import logging
import re
import sys


_MODEL_CONFIG_START = re.compile(r"Building .* with \{$")
_LOGGER = logging.getLogger("smoke_log_filter")


def _init_logger() -> None:
    _LOGGER.setLevel(logging.INFO)
    _LOGGER.propagate = False
    if _LOGGER.handlers:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    _LOGGER.addHandler(handler)


def _emit(line: str) -> None:
    _LOGGER.info("%s", line)


def _brace_delta(line: str) -> int:
    return line.count("{") - line.count("}")


def main() -> None:
    _init_logger()

    skipping_model_config = False
    model_config_depth = 0
    skipping_model_definition = False
    seen_model_definition_body = False

    for raw_line in sys.stdin:
        line = raw_line.rstrip("\n")

        if skipping_model_config:
            model_config_depth += _brace_delta(line)
            if model_config_depth <= 0:
                skipping_model_config = False
            continue

        if skipping_model_definition:
            if line.strip():
                seen_model_definition_body = True
            elif seen_model_definition_body:
                skipping_model_definition = False
            continue

        if "Model definition after conversion:" in line:
            prefix = line.split("Model definition after conversion:", 1)[0]
            _emit(
                f"{prefix}Model definition after conversion suppressed for smoke test."
            )
            skipping_model_definition = True
            seen_model_definition_body = False
            continue

        if _MODEL_CONFIG_START.search(line):
            _emit(
                re.sub(
                    r" with \{$",
                    " (model config suppressed for smoke test)",
                    line,
                )
            )
            skipping_model_config = True
            model_config_depth = _brace_delta(line)
            if model_config_depth <= 0:
                skipping_model_config = False
            continue

        _emit(line)


if __name__ == "__main__":
    main()
