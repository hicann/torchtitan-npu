# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""NPU deterministic env setup driven by ``--debug.deterministic``.

Upstream owns RNG seeding and ``torch.use_deterministic_algorithms``. This
module only injects NPU env vars that must be set before distributed init.
"""

import os

from torchtitan.config import DebugConfig
from torchtitan.tools.logging import logger

_DETERMINISTIC_ENV: dict[str, str] = {
    "HCCL_DETERMINISTIC": "true",
    "CLOSE_MATMUL_K_SHIFT": "1",
}


def _set_env_with_warning(name: str, value: str) -> None:
    """Set env value and warn when overriding a conflicting user value."""
    existing = os.environ.get(name)
    if existing is not None and existing != value:
        logger.warning(
            "Overriding existing env %s=%r with %r for deterministic training "
            "(requested by --debug.deterministic).",
            name,
            existing,
            value,
        )
    os.environ[name] = value


def setup_npu_deterministic_env(debug_config: DebugConfig) -> None:
    """Inject NPU deterministic env vars before distributed init."""
    if not debug_config.deterministic:
        return

    for name, value in _DETERMINISTIC_ENV.items():
        _set_env_with_warning(name, value)

    logger.info(
        "NPU deterministic env enabled (%s) via --debug.deterministic.",
        ", ".join(_DETERMINISTIC_ENV),
    )
