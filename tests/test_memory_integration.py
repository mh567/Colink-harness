"""Tests for Memory integration into HarnessAgent."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from octop_memory import Memory

from octop_harness.agent import HarnessAgent
from octop_harness.config import HarnessAgentConfig, ModelConfig, ProviderConfig


@pytest.fixture
def cfg(tmp_path: Path) -> HarnessAgentConfig:
    return HarnessAgentConfig(
        name="test-agent",
        workspace_dir=tmp_path,
        backend={"type": "filesystem", "root_dir": str(tmp_path), "virtual_mode": True},
        providers=[
            ProviderConfig(id="p", base_url="https://x", api_key="k", models=[ModelConfig(id="m")]),
        ],
        memory_enabled=True,
        memory_backend={"type": "sqlite", "db_path": str(tmp_path / "mem.sqlite")},
        checkpointer=False,
    )


@pytest.fixture
def agent(cfg: HarnessAgentConfig) -> Iterator[HarnessAgent]:
    with (
        patch("langchain_openai.ChatOpenAI", return_value=MagicMock()),
        patch("deepagents.create_deep_agent", return_value=MagicMock()),
    ):
        octop_harness = HarnessAgent(cfg)
    yield octop_harness
    octop_harness.close()


class TestMemoryIntegration:
    def test_agent_has_memory_property(self, agent: HarnessAgent) -> None:
        assert isinstance(agent.memory, Memory)

    def test_live_memory_access_revokes_prebuilt_tools_without_replacing_checkpointer(
        self, cfg: HarnessAgentConfig
    ) -> None:
        with (
            patch("langchain_openai.ChatOpenAI", return_value=MagicMock()),
            patch("deepagents.create_deep_agent", return_value=MagicMock()),
            HarnessAgent(replace(cfg, checkpointer=None)) as agent,
        ):
            tools = {tool.name: tool for tool in agent._memory_runtime.build_tools()}
            checkpointer = agent.checkpointer
            agent.memory.create_atom("The preferred font is Garamond", entity_name="Fonts", kind="Preference")
            assert "Garamond" in tools["memory_search"].invoke({"query": "Garamond"})

            agent.set_memory_access(False)

            assert agent.checkpointer is checkpointer
            assert "Garamond" not in tools["memory_search"].invoke({"query": "Garamond"})
            tools["memory_save"].invoke({"content": "The preferred font is Courier", "topic": "Fonts"})
            assert not agent.memory.search_atoms("Courier")

            agent.set_memory_access(True)
            assert "Garamond" in tools["memory_search"].invoke({"query": "Garamond"})

    def test_startup_memory_access_stays_off_after_config_round_trip(self, cfg: HarnessAgentConfig) -> None:
        wire = cfg.to_dict()
        wire["memory_access_enabled"] = False
        reloaded = HarnessAgentConfig.from_dict(wire)

        assert reloaded.to_dict()["memory_access_enabled"] is False

    def test_memory_tools_registered(self, cfg: HarnessAgentConfig) -> None:
        captured: dict[str, Any] = {}

        def fake_create(**kwargs: Any) -> Any:
            captured.update(kwargs)
            return MagicMock()

        with (
            patch("langchain_openai.ChatOpenAI", return_value=MagicMock()),
            patch("deepagents.create_deep_agent", side_effect=fake_create),
        ):
            HarnessAgent(cfg)

        tool_names = [getattr(t, "name", "") for t in captured["tools"]]
        # Explicit remember requests persist through the canonical service.
        assert "memory_save" in tool_names
        assert "memory_search" in tool_names
        assert "memory_get" in tool_names
        assert "memory_store" not in tool_names
        assert "memory_recall" not in tool_names

    def test_memory_middleware_registered(self, cfg: HarnessAgentConfig) -> None:
        captured: dict[str, Any] = {}

        def fake_create(**kwargs: Any) -> Any:
            captured.update(kwargs)
            return MagicMock()

        with (
            patch("langchain_openai.ChatOpenAI", return_value=MagicMock()),
            patch("deepagents.create_deep_agent", side_effect=fake_create),
        ):
            HarnessAgent(cfg)

        from octop_harness.middleware.memory import MemoryMiddleware

        mw_types = [type(m) for m in captured["middleware"]]
        assert MemoryMiddleware in mw_types

    def test_memory_disabled_uses_jsonl_only_middleware(self, tmp_path: Path) -> None:
        """When memory is off but session_log is on, MemoryMiddleware
        is still installed in JSONL-only mode."""
        cfg = HarnessAgentConfig(
            workspace_dir=tmp_path,
            backend={"type": "filesystem", "root_dir": str(tmp_path), "virtual_mode": True},
            providers=[ProviderConfig(id="p", base_url="https://x", api_key="k", models=[ModelConfig(id="m")])],
            memory_enabled=False,
        )
        captured: dict[str, Any] = {}

        def fake_create(**kwargs: Any) -> Any:
            captured.update(kwargs)
            return MagicMock()

        with (
            patch("langchain_openai.ChatOpenAI", return_value=MagicMock()),
            patch("deepagents.create_deep_agent", side_effect=fake_create),
        ):
            agent = HarnessAgent(cfg)

        from octop_harness.middleware.memory import MemoryMiddleware

        mw_instances = [m for m in captured["middleware"] if isinstance(m, MemoryMiddleware)]
        assert len(mw_instances) == 1, "expected exactly one MemoryMiddleware"
        assert mw_instances[0]._service is None
        assert mw_instances[0]._jsonl_enabled is True
        assert agent.memory is None

    def test_default_sqlite_memory_db_is_workspace_scoped(self, tmp_path: Path) -> None:
        cfg = HarnessAgentConfig(
            name="test-agent",
            workspace_dir=tmp_path,
            backend={"type": "filesystem", "root_dir": str(tmp_path), "virtual_mode": True},
            providers=[
                ProviderConfig(id="p", base_url="https://x", api_key="k", models=[ModelConfig(id="m")]),
            ],
            memory_enabled=True,
            memory_backend="sqlite",
        )

        with (
            patch("langchain_openai.ChatOpenAI", return_value=MagicMock()),
            patch("deepagents.create_deep_agent", return_value=MagicMock()),
        ):
            agent = HarnessAgent(cfg)

        assert agent.memory is not None
        assert Path(agent.memory.backend._db_path) == tmp_path / "memory.sqlite"
