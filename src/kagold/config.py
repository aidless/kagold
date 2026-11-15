"""kagold 配置与路径解析。

设计原则：
- 所有路径由这里统一给出，工具层不允许自己拼路径。
- LLM provider 复用 ~/.dsh 的凭据库，避免再维护一份 key。
- 预算/守卫参数集中在这里，agent 无法通过 prompt 覆盖。
"""
from __future__ import annotations

import os
import shutil
import socket
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

# kagold/src/kagold/config.py -> kagold/
PROJECT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT / "src"
RUNS_DIR = PROJECT / "runs"
LOG_DIR = PROJECT / "logs"


def _read_yaml(path: Path) -> dict:
    if yaml is None or not path.exists():
        return {}
    try:
        return yaml.safe_load(path.read_text()) or {}
    except Exception:
        return {}


def _read_env_file(path: Path) -> dict:
    """解析 ~/.dsh/.env 这类 KEY=VALUE 文件。"""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


@dataclass
class Provider:
    name: str
    base_url: str
    api_key: str
    model: str
    source: str = ""


def _dsh_providers() -> dict[str, dict]:
    cfg = _read_yaml(Path.home() / ".dsh" / "settings.yaml.imported")
    provs = ((cfg.get("llm-pi-ai") or {}).get("providers")) or {}
    return {k: v for k, v in provs.items() if isinstance(v, dict)}


def _dsh_secret(env_name: str) -> str | None:
    """按 ~/.dsh/.env -> 环境变量 -> .credentials.yaml refs 的顺序取密钥。"""
    val = _read_env_file(Path.home() / ".dsh" / ".env").get(env_name)
    if val:
        return val
    if os.environ.get(env_name):
        return os.environ[env_name]
    cred = _read_yaml(Path.home() / ".dsh" / ".credentials.yaml")
    refs = cred.get("refs") or {}
    if isinstance(refs, dict) and refs.get(env_name):
        return str(refs[env_name])
    return None


def resolve_provider(name: str | None = None, model: str | None = None) -> Provider:
    """解析一个 OpenAI 兼容 provider。优先用 kagold.yaml 的 llm.provider。"""
    provs = _dsh_providers()
    name = name or os.environ.get("KAGOLD_PROVIDER") or "huo"
    if name not in provs:
        # 允许直接给 base_url 的模式
        base = os.environ.get("KAGOLD_BASE_URL")
        key = os.environ.get("KAGOLD_API_KEY") or ""
        if base and key:
            return Provider(name="env", base_url=base, api_key=key,
                            model=model or os.environ.get("KAGOLD_MODEL", "deepseek-v4.1-flash"),
                            source="env")
        raise SystemExit(f"未知 provider={name!r}；可用：{sorted(provs)}  或设置 KAGOLD_BASE_URL/KAGOLD_API_KEY")

    p = provs[name]
    base_url = os.environ.get("KAGOLD_BASE_URL") or p.get("baseURL") or ""
    api_key_env = p.get("apiKeyEnv") or ""
    api_key = os.environ.get("KAGOLD_API_KEY") or _dsh_secret(api_key_env) or ""
    models = p.get("models") or []
    ids = [m.get("id") if isinstance(m, dict) else m for m in models]
    chosen = model or os.environ.get("KAGOLD_MODEL") or (ids[0] if ids else "")
    if not api_key:
        raise SystemExit(f"provider={name} 缺少密钥（env {api_key_env}），检查 ~/.dsh/.env 或 .credentials.yaml")
    if not base_url:
        raise SystemExit(f"provider={name} 缺少 baseURL")
    return Provider(name=name, base_url=base_url.rstrip("/"), api_key=api_key,
                    model=chosen, source=f"dsh:{api_key_env}")


@dataclass
class Settings:
    # 预算（agent 改不动）
    max_steps: int = 140
    max_total_steps: int = 600
    max_total_hours: float = 24.0
    max_tokens_in: int = 4_000_000
    max_tokens_out: int = 500_000
    max_hours: float = 8.0
    max_submissions: int = 6
    max_shell_seconds: int = 1800
    # 守卫
    min_free_gb: float = 3.0          # 下载前磁盘守卫
    max_tool_output: int = 6000        # 单个工具结果回灌给模型的字符上限
    temperature: float = 0.25
    # 行为
    allow_accept_rules: bool = False   # 是否允许自动接受竞赛规则（法律动作，默认关）
    allow_public_notebooks: bool = False
    llm_provider: str = "huo"
    llm_model: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        cfg = _read_yaml(PROJECT / "kagold.yaml")
        llm = cfg.pop("llm", {}) if isinstance(cfg, dict) else {}
        self.llm_provider = os.environ.get("KAGOLD_PROVIDER", llm.get("provider", self.llm_provider))
        self.llm_model = os.environ.get("KAGOLD_MODEL", llm.get("model", self.llm_model))
        for k, v in (cfg or {}).items():
            if hasattr(self, k) and v is not None:
                setattr(self, k, v)
        self.extra = cfg or {}

    def as_dict(self) -> dict:
        return asdict(self)


def settings(**overrides) -> Settings:
    s = Settings()
    for k, v in overrides.items():
        if v is not None and hasattr(s, k):
            setattr(s, k, v)
    return s


def slug_of(ref: str) -> str:
    """'https://www.kaggle.com/competitions/foo' -> 'foo'"""
    ref = (ref or "").strip().rstrip("/")
    if "competitions/" in ref:
        return ref.split("competitions/")[-1].split("?")[0].split("/")[0]
    return ref.split("/")[-1]


def run_root(comp: str) -> Path:
    return RUNS_DIR / slug_of(comp)


def ensure_run_dirs(comp: str) -> Path:
    root = run_root(comp)
    for sub in ("input", "src", "oof", "sub", "logs", "notebooks"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root


def free_gb(path: Path | str = "/") -> float:
    try:
        return shutil.disk_usage(str(path)).free / (1024 ** 3)
    except Exception:
        return 999.0


def disk_guard(min_free_gb: float, need_gb: float = 0.0, path: Path | str = "/") -> None:
    have = free_gb(path)
    if have - need_gb < min_free_gb:
        raise RuntimeError(
            f"磁盘守卫：{path} 仅剩 {have:.2f}GB，本次需要约 {need_gb:.2f}GB，"
            f"低于阈值 {min_free_gb}GB，拒绝执行。请先清理（runs/*/input 下旧数据）。"
        )


def host_info() -> dict:
    import multiprocessing
    return {
        "host": socket.gethostname(),
        "cpus": multiprocessing.cpu_count(),
        "free_gb": round(free_gb("/"), 2),
        "gpu": "none",
    }
