# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import pickle
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch

from torchtitan_npu.patches.quantization import quantize
from torchtitan_npu.patches.torch import clip_grad


def test_custom_checkpoint_config_exposes_writer_controls():
    from torchtitan_npu.config.custom_config import Checkpoint, JobConfig

    checkpoint = Checkpoint()
    job_config = JobConfig()

    assert checkpoint.sync_files is True
    assert checkpoint.drop_page_cache_after_save is False
    assert job_config.checkpoint.sync_files is True


def test_drop_page_cache_for_path_uses_posix_fadvise(monkeypatch, tmp_path):
    from torchtitan_npu.patches.torch import checkpoint

    data_file = tmp_path / "data.distcp"
    data_file.write_bytes(b"checkpoint")
    calls = []

    def fake_posix_fadvise(fd, offset, length, advice):
        calls.append((fd, offset, length, advice))

    monkeypatch.setattr(checkpoint.os, "posix_fadvise", fake_posix_fadvise)

    checkpoint.drop_page_cache_for_path(tmp_path)

    assert calls
    assert calls[0][1:] == (0, 0, checkpoint.os.POSIX_FADV_DONTNEED)


@pytest.mark.parametrize("async_mode", ["disabled", "async", "async_with_pinned_mem"])
def test_checkpoint_manager_dcp_save_builds_writer_for_all_async_modes(
    monkeypatch, tmp_path, async_mode
):
    from torch.distributed.checkpoint.filesystem import FileSystemWriter
    from torchtitan.components.checkpoint import AsyncMode

    from torchtitan_npu.patches.torch import checkpoint

    mode = AsyncMode(async_mode)
    stager = object()
    captured = {}

    def fake_dcp_save(*args, storage_writer=None, **kwargs):
        captured["writer"] = storage_writer
        captured["kwargs"] = kwargs
        return "saved"

    def fake_dcp_async_save(*args, storage_writer=None, async_stager=None, **kwargs):
        captured["writer"] = storage_writer
        captured["async_stager"] = async_stager
        captured["kwargs"] = kwargs
        return "future"

    manager = SimpleNamespace(
        pg=None,
        stager=stager,
        _npu_checkpoint_sync_files=False,
        _npu_checkpoint_drop_page_cache_after_save=True,
    )

    monkeypatch.setattr(checkpoint.dcp, "save", fake_dcp_save)
    monkeypatch.setattr(checkpoint.dcp, "async_save", fake_dcp_async_save)

    result = checkpoint._patched_checkpoint_manager_dcp_save(
        manager, {}, str(tmp_path / "checkpoint"), mode
    )
    writer = captured["writer"]

    assert isinstance(writer, FileSystemWriter)
    assert writer.sync_files is False
    writer_with_options: Any = writer
    assert writer_with_options._npu_drop_page_cache_after_save is True

    restored_writer = pickle.loads(pickle.dumps(writer))
    restored_writer_with_options: Any = restored_writer
    assert restored_writer.sync_files is False
    assert restored_writer_with_options._npu_drop_page_cache_after_save is True

    if mode == AsyncMode.DISABLED:
        assert result == "saved"
        assert "async_stager" not in captured
    elif mode == AsyncMode.ASYNC:
        assert result == "future"
        assert captured["async_stager"] is None
    else:
        assert result == "future"
        assert captured["async_stager"] is stager
        assert (
            captured["kwargs"]["async_checkpointer_type"]
            is checkpoint.AsyncCheckpointerType.PROCESS
        )


def test_register_quantize_module_handler_registers_handler():
    class DummyConfig:
        pass

    def handler(module, config):
        return module

    handler_registry = {}
    with patch.object(quantize, "_QUANTIZE_CONFIG_HANDLER", handler_registry):
        decorated = quantize.register_quantize_module_handler(DummyConfig)(handler)
        assert decorated is handler
        assert handler_registry.get(DummyConfig) is handler


def test_group_dtensors_by_layout_groups_non_dtensors_together():
    tensor_a = torch.randn(2, 2)
    tensor_b = torch.randn(2, 2)

    grouped = clip_grad.group_dtensors_by_layout([tensor_a, tensor_b])

    assert len(grouped) == 1
    assert ("non_dtensor", None) in grouped
    assert grouped[("non_dtensor", None)] == [tensor_a, tensor_b]


def test_group_dtensors_by_layout_handles_empty_input():
    grouped = clip_grad.group_dtensors_by_layout([])

    assert grouped == {}


def test_register_quantize_module_handler_overrides_existing_handler():
    class DummyConfig:
        pass

    def old_handler(module, config):
        return module

    def new_handler(module, config):
        return module

    handler_registry = {DummyConfig: old_handler}
    with patch.object(quantize, "_QUANTIZE_CONFIG_HANDLER", handler_registry):
        quantize.register_quantize_module_handler(DummyConfig)(new_handler)
        assert handler_registry[DummyConfig] is new_handler
