# SPDX-FileCopyrightText: Copyright (c) 2026 BlaiseAI / ai-blaise. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

if os.environ.get("MEGATRON_CRIU_ENABLE") == "1":
    try:
        from megatron_criu_hooks import install

        install()
    except Exception as exc:
        raise RuntimeError("failed to install Megatron CRIU hooks") from exc
