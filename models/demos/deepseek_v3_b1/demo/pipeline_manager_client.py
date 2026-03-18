# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from models.demos.deepseek_v3_b1.micro_ops.pipeline_block.op import HostSocketDescriptorBundle


@dataclass(frozen=True)
class PipelineManagerRequest:
    request_id: str
    prompt_token_ids: list[int]
    max_new_tokens: int
    eos_token_id: int | None = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def resolve_pipeline_manager_binary(manager_binary: str | Path | None = None) -> Path:
    candidates: list[Path] = []
    if manager_binary is not None:
        candidates.append(Path(manager_binary))

    env_binary = os.environ.get("TT_DEEPSEEK_PIPELINE_MANAGER_BIN")
    if env_binary:
        candidates.append(Path(env_binary))

    repo_root = _repo_root()
    candidates.extend(
        [
            repo_root / "build" / "tools" / "deepseek_v3_b1_pipeline_manager",
            repo_root / "build_Release" / "tools" / "deepseek_v3_b1_pipeline_manager",
            repo_root / "build_Debug" / "tools" / "deepseek_v3_b1_pipeline_manager",
            repo_root / "cmake-build-release" / "tools" / "deepseek_v3_b1_pipeline_manager",
            repo_root / "cmake-build-debug" / "tools" / "deepseek_v3_b1_pipeline_manager",
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not locate deepseek_v3_b1_pipeline_manager. "
        "Set TT_DEEPSEEK_PIPELINE_MANAGER_BIN or pass --pipeline-manager-bin."
    )


class PipelineManagerClient:
    def __init__(self, process: subprocess.Popen[str]) -> None:
        self._process = process

    @classmethod
    def launch(
        cls,
        descriptors: HostSocketDescriptorBundle,
        *,
        manager_binary: str | Path | None = None,
        connect_timeout_ms: int = 30000,
    ) -> PipelineManagerClient:
        binary = resolve_pipeline_manager_binary(manager_binary)
        process = subprocess.Popen(
            [
                str(binary),
                "--mode",
                "manager",
                "--h2d-socket-id",
                descriptors.h2d_socket_id,
                "--d2h-socket-id",
                descriptors.d2h_socket_id,
                "--page-size-bytes",
                str(descriptors.page_size_bytes),
                "--connect-timeout-ms",
                str(connect_timeout_ms),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        client = cls(process)
        client._expect_ready()
        return client

    def _expect_ready(self) -> None:
        kind, fields = self._read_event()
        if kind != "READY" or fields != ["manager"]:
            raise RuntimeError(f"Unexpected manager startup response: {kind} {fields}")

    def _read_event(self) -> tuple[str, list[str]]:
        assert self._process.stdout is not None
        line = self._process.stdout.readline()
        if line == "":
            stderr = ""
            if self._process.stderr is not None:
                stderr = self._process.stderr.read()
            raise RuntimeError(f"Pipeline manager exited unexpectedly. stderr={stderr.strip()}")

        parts = line.rstrip("\n\r").split("\t")
        if not parts or not parts[0]:
            raise RuntimeError("Pipeline manager returned an empty protocol line")
        return parts[0], parts[1:]

    def run_request(
        self,
        request: PipelineManagerRequest,
        *,
        on_token: Callable[[int], None] | None = None,
        return_generated_tokens: bool = False,
    ) -> list[int] | None:
        assert self._process.stdin is not None
        eos_token_id = -1 if request.eos_token_id is None else request.eos_token_id
        prompt_tokens = ",".join(str(token_id) for token_id in request.prompt_token_ids)
        self._process.stdin.write(
            f"SUBMIT\t{request.request_id}\t{request.max_new_tokens}\t{eos_token_id}\t{prompt_tokens}\n"
        )
        self._process.stdin.flush()

        generated_tokens: list[int] = []
        while True:
            kind, fields = self._read_event()
            if kind == "TOKEN":
                if len(fields) != 3 or fields[0] != request.request_id:
                    raise RuntimeError(f"Unexpected TOKEN event: {fields}")
                token_id = int(fields[2])
                generated_tokens.append(token_id)
                if on_token is not None:
                    on_token(token_id)
                continue

            if kind == "COMPLETE":
                if len(fields) < 3 or fields[0] != request.request_id:
                    raise RuntimeError(f"Unexpected COMPLETE event: {fields}")
                return generated_tokens if return_generated_tokens else None

            if kind == "ERROR":
                if len(fields) < 2:
                    raise RuntimeError("Pipeline manager returned a malformed ERROR event")
                if fields[0] == request.request_id or fields[0] == "manager":
                    raise RuntimeError(fields[1])
                continue

            raise RuntimeError(f"Unexpected event from pipeline manager: {kind} {fields}")

    def close(self) -> None:
        if self._process.poll() is not None:
            return

        assert self._process.stdin is not None
        self._process.stdin.write("SHUTDOWN\n")
        self._process.stdin.flush()
        self._process.stdin.close()
        self._process.wait(timeout=30)

    def __enter__(self) -> PipelineManagerClient:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
