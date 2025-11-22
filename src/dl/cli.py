"""
dl.cli
=================

Command-line entry point for the :mod:`dl` download dispatcher.

Features:
- Choose download tool based on URL patterns and (optionally) MIME type / size.
- Supports external tools like yt-dlp, gallery-dl, aria2c, curl, wget.
- Configurable via a TOML or JSON config file.
- Auto-generates a default config file if none is found.
- Fallback pure Python HTTP downloader when no external tool is available.

Default config location (if none exists yet):
  $XDG_CONFIG_HOME/dl/config.toml  (if tomllib available, Python 3.11+)
  else $XDG_CONFIG_HOME/dl/config.json
If XDG_CONFIG_HOME is not set, falls back to ~/.config/dl/...

You can override via:
  --config PATH
or:
  DL_CONFIG=/path/to/config dl ...
"""

from __future__ import annotations

import argparse
import atexit
import dataclasses
import datetime
import functools
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import textwrap
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, TextIO, Tuple

from . import __version__

# Try stdlib TOML parser (Python 3.11+)
try:  # pragma: no cover - version dependent
    import tomllib  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    tomllib = None  # type: ignore[assignment]


@dataclass
class ProbeResult:
    size: Optional[int] = None
    mime: Optional[str] = None


@dataclass
class Rule:
    name: str
    match: str
    tool: str
    args: List[str] = field(default_factory=list)
    when_minsize: Optional[int] = None
    when_maxsize: Optional[int] = None
    when_mime: Optional[str] = None
    profiles: List[str] = field(default_factory=list)

    def matches_profile(self, profile: str) -> bool:
        if not self.profiles:
            # No profile means rule applies everywhere
            return True
        return profile in self.profiles

    def matches_url(self, url: str) -> bool:
        try:
            return re.search(self.match, url) is not None
        except re.error:
            return False

    def matches_probe(self, probe: Optional[ProbeResult]) -> bool:
        if probe is None:
            # If we have constraints that require probe data but do not have it,
            # fail them conservatively.
            if self.when_minsize is not None or self.when_maxsize is not None or self.when_mime:
                return False
            return True

        if self.when_minsize is not None and probe.size is not None:
            if probe.size < self.when_minsize:
                return False

        if self.when_maxsize is not None and probe.size is not None:
            if probe.size > self.when_maxsize:
                return False

        if self.when_mime:
            if not probe.mime:
                return False
            if not mime_matches(self.when_mime, probe.mime):
                return False

        return True


DEFAULT_USER_AGENT = "dl/0.1 (Python urllib)"


def env_flag(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    return value.strip().lower() not in {"", "0", "false", "no"}


INSTALL_HINTS_SUPPRESS = env_flag("DL_NO_INSTALL_HINTS")


DEFAULT_CONFIG: Dict[str, Any] = {
    "default_tool": None,  # autodetect
    "default_args": [],
    "rules": [
        {
            "name": "video_common_sites_yt_dlp",
            "match": r"^(https?://)?(www\.)?(youtube\.com|youtu\.be|twitch\.tv|vimeo\.com|bilibili\.com)/",
            "tool": "yt-dlp",
            "args": ["-f", "bv*+ba/b"],
            "profiles": ["default", "video"],
        },
        {
            "name": "audio_only_profile",
            "match": r"^(https?://)?(www\.)?(youtube\.com|youtu\.be)/",
            "tool": "yt-dlp",
            "args": ["-x", "--audio-format", "mp3"],
            "profiles": ["audio"],
        },
        {
            "name": "image_galleries_gallery_dl",
            "match": r"^(https?://)?(www\.)?(danbooru\.donmai\.us|gelbooru\.com|pixiv\.net|twitter\.com|x\.com)/",
            "tool": "gallery-dl",
            "args": [],
            "profiles": ["default", "images", "gallery"],
        },
        {
            "name": "magnet_links_aria2",
            "match": r"^magnet:\?",
            "tool": "aria2c",
            "args": [],
            "profiles": ["default", "torrents"],
        },
        {
            "name": "large_http_files_aria2",
            "match": r"^https?://",
            "tool": "aria2c",
            "args": ["-x", "16", "-s", "16"],
            "when_minsize": "50M",
            "profiles": ["default", "large"],
        },
        {
            "name": "http_generic_curl",
            "match": r"^https?://",
            "tool": "curl",
            "args": ["-L", "-O"],
            "profiles": ["default"],
        },
    ],
}

INSTALL_HINTS: Dict[str, Dict[str, str]] = {
    "yt-dlp": {
        "pip": "python3 -m pip install yt-dlp",
        "pipx": "pipx install yt-dlp",
        "homebrew": "brew install yt-dlp",
        "apt": "sudo apt install yt-dlp",
        "dnf": "sudo dnf install yt-dlp",
    },
    "gallery-dl": {
        "pip": "python3 -m pip install gallery-dl",
        "pipx": "pipx install gallery-dl",
        "homebrew": "brew install gallery-dl",
    },
    "aria2c": {
        "homebrew": "brew install aria2",
        "apt": "sudo apt install aria2",
        "dnf": "sudo dnf install aria2",
    },
    "curl": {
        "homebrew": "brew install curl",
        "apt": "sudo apt install curl",
        "dnf": "sudo dnf install curl",
    },
    "wget": {
        "homebrew": "brew install wget",
        "apt": "sudo apt install wget",
        "dnf": "sudo dnf install wget",
    },
}

INSTALLER_CANDIDATES: Dict[str, Tuple[str, ...]] = {
    "pip": ("pip3", "pip"),
    "pipx": ("pipx",),
    "homebrew": ("brew",),
    "apt": ("apt", "apt-get"),
    "dnf": ("dnf", "yum"),
}


LOG_SINK: Optional["LogEmitter"] = None
_LOG_SHUTDOWN_REGISTERED = False

LOG_THEMES: Dict[str, Dict[str, str]] = {
    "default": {
        "INFO": "\x1b[32m",  # green
        "DEBUG": "\x1b[36m",  # cyan
        "TRACE": "\x1b[90m",  # gray
        "WARNING": "\x1b[33m",  # yellow
        "ERROR": "\x1b[31m",  # red
        "RESET": "\x1b[0m",
    },
    "vibrant": {
        "INFO": "\x1b[1;34m",  # bold blue
        "DEBUG": "\x1b[1;35m",  # magenta
        "TRACE": "\x1b[38;5;244m",  # dim gray
        "WARNING": "\x1b[1;33m",  # bold yellow
        "ERROR": "\x1b[1;31m",  # bold red
        "RESET": "\x1b[0m",
    },
}


def resolve_theme(name: str) -> Dict[str, str]:
    return LOG_THEMES.get(name, LOG_THEMES["default"])


def should_colorize(mode: str, fmt: str) -> bool:
    if fmt == "json":
        return False
    if mode == "always":
        return True
    if mode == "never":
        return False
    return sys.stderr.isatty()


@dataclass
class LogEmitter:
    verbose: int
    fmt: str
    timestamps: bool
    stream: TextIO = sys.stderr
    file_handle: Optional[TextIO] = None
    name: str = "dl"
    color: bool = False
    theme: Dict[str, str] = field(default_factory=dict)

    def close(self) -> None:
        if self.file_handle:
            try:
                self.file_handle.close()
            finally:
                self.file_handle = None

    def _severity_for(self, level: int) -> str:
        if level <= 0:
            return "INFO"
        if level == 1:
            return "DEBUG"
        return "TRACE"

    def log(self, level: int, msg: str, severity: Optional[str] = None) -> None:
        sev = (severity or self._severity_for(level)).upper()
        if sev not in {"ERROR", "WARNING"} and self.verbose < level:
            return
        record: Dict[str, str] = {
            "name": self.name,
            "level": sev,
            "message": msg,
        }
        if self.timestamps:
            record["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        line = self._format(record)
        if self.color and self.fmt == "plain":
            line = self._apply_color(line, sev)
        self.stream.write(line)
        self.stream.flush()
        if self.file_handle:
            self.file_handle.write(line)
            self.file_handle.flush()

    def _format(self, record: Dict[str, str]) -> str:
        if self.fmt == "json":
            payload = {k: v for k, v in record.items()}
            return json.dumps(payload, ensure_ascii=False) + "\n"

        parts: List[str] = []
        timestamp = record.get("timestamp")
        if timestamp:
            parts.append(timestamp)
        parts.append(f"[{record['name']}]")
        level = record.get("level")
        if level:
            parts.append(f"[{level}]")
        parts.append(record.get("message", ""))
        return " ".join(parts) + "\n"

    def _apply_color(self, line: str, severity: str) -> str:
        color_code = self.theme.get(severity) or self.theme.get("INFO")
        reset = self.theme.get("RESET", "\x1b[0m")
        if not color_code:
            return line
        if line.endswith("\n"):
            content = line[:-1]
            return f"{color_code}{content}{reset}\n"
        return f"{color_code}{line}{reset}"

    def debug(self, level: int, msg: str) -> None:
        self.log(level, msg)

    def info(self, msg: str) -> None:
        self.log(0, msg, severity="INFO")

    def warning(self, msg: str) -> None:
        self.log(0, msg, severity="WARNING")

    def error(self, msg: str) -> None:
        self.log(0, msg, severity="ERROR")


def configure_logging(
    verbose: int,
    fmt: str,
    timestamps: bool,
    log_path: Optional[str],
    color: bool,
    theme_name: str,
) -> None:
    global LOG_SINK, _LOG_SHUTDOWN_REGISTERED
    file_handle: Optional[TextIO] = None
    if log_path:
        try:
            file_handle = open(log_path, "a", encoding="utf-8")
        except OSError as exc:
            sys.stderr.write(f"[dl] Failed to open log file {log_path!r}: {exc!r}\n")
            sys.stderr.flush()
            file_handle = None
    LOG_SINK = LogEmitter(
        verbose=verbose,
        fmt=fmt,
        timestamps=timestamps,
        stream=sys.stderr,
        file_handle=file_handle,
        color=color if fmt == "plain" else False,
        theme=resolve_theme(theme_name),
    )
    if not _LOG_SHUTDOWN_REGISTERED:
        atexit.register(shutdown_logging)
        _LOG_SHUTDOWN_REGISTERED = True


def shutdown_logging() -> None:
    global LOG_SINK
    if LOG_SINK:
        LOG_SINK.close()
    LOG_SINK = None


@functools.lru_cache(maxsize=1)
def detect_installers() -> List[str]:
    """Return a list of installer keys that are available on this system."""
    available: List[str] = []
    for key, candidates in INSTALLER_CANDIDATES.items():
        for candidate in candidates:
            if shutil.which(candidate):
                available.append(key)
                break
    return available


def get_install_instructions(tool: str) -> List[Tuple[str, str]]:
    """Return human friendly commands for installing *tool*."""
    hints = INSTALL_HINTS.get(tool)
    if not hints:
        return []

    available = detect_installers()
    ordered_keys = [key for key in available if key in hints]
    if not ordered_keys:
        ordered_keys = list(hints.keys())
    return [(key, hints[key]) for key in ordered_keys]


def show_install_instructions(tool: str) -> None:
    """Print installation hints for a missing *tool*."""
    if INSTALL_HINTS_SUPPRESS:
        return
    instructions = get_install_instructions(tool)
    if not instructions:
        return
    log_info(f"Installation hints for missing tool {tool!r}:")
    for manager, command in instructions:
        log_info(f"  {manager:10}: {command}")


def debug_print(verbose: int, level: int, msg: str) -> None:
    if LOG_SINK:
        LOG_SINK.debug(level, msg)
        return
    if verbose >= level:
        sys.stderr.write(f"[dl] {msg}\n")
        sys.stderr.flush()


def suppress_install_hints() -> None:
    global INSTALL_HINTS_SUPPRESS
    INSTALL_HINTS_SUPPRESS = True


def log_info(msg: str) -> None:
    if LOG_SINK:
        LOG_SINK.info(msg)
    else:
        sys.stderr.write(f"[dl] {msg}\n")
        sys.stderr.flush()


def log_warning(msg: str) -> None:
    if LOG_SINK:
        LOG_SINK.warning(msg)
    else:
        sys.stderr.write(f"[dl][WARN] {msg}\n")
        sys.stderr.flush()


def log_error(msg: str) -> None:
    if LOG_SINK:
        LOG_SINK.error(msg)
    else:
        sys.stderr.write(f"[dl][ERROR] {msg}\n")
        sys.stderr.flush()


def human_size(num: Optional[int]) -> str:
    if num is None:
        return "unknown"
    step = 1024.0
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if num < step:
            return f"{num:.1f}{unit}"
        num /= step
    return f"{num:.1f}PiB"


def parse_size(s: Any) -> Optional[int]:
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return int(s)
    if not isinstance(s, str):
        return None

    s = s.strip()
    if not s:
        return None

    m = re.match(r"^(\d+(?:\.\d+)?)([KkMmGgTtPp]?)[Bb]?$", s)
    if not m:
        return None
    value = float(m.group(1))
    unit = m.group(2).upper()
    multipliers = {
        "": 1,
        "K": 1024,
        "M": 1024**2,
        "G": 1024**3,
        "T": 1024**4,
        "P": 1024**5,
    }
    return int(value * multipliers.get(unit, 1))


def mime_matches(pattern: str, mime: str) -> bool:
    pattern = pattern.lower().strip()
    mime = mime.lower().strip()

    if pattern == "*":
        return True
    if "/" not in pattern:
        # Treat "video" as "video/*"
        pattern = pattern + "/*"
    if "/" not in mime:
        return False

    p_main, p_sub = pattern.split("/", 1)
    m_main, m_sub = mime.split("/", 1)

    if p_main != "*" and p_main != m_main:
        return False
    if p_sub != "*" and p_sub != m_sub:
        return False
    return True


def detect_default_tool() -> str:
    for candidate in ("curl", "wget"):
        if shutil.which(candidate):
            return candidate
    # Fallback to built in Python downloader
    return "python-http"


@dataclass
class EffectiveConfig:
    """Runtime-ready configuration object derived from raw config data."""

    rules: List[Rule]
    default_tool: str
    default_args: List[str]

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "EffectiveConfig":
        cfg_default_tool = config.get("default_tool") or detect_default_tool()
        cfg_default_args = list(config.get("default_args") or [])

        rules_cfg = config.get("rules") or []
        rules: List[Rule] = []

        for item in rules_cfg:
            if not isinstance(item, dict):
                continue
            when_minsize = parse_size(item.get("when_minsize"))
            when_maxsize = parse_size(item.get("when_maxsize"))
            profiles = item.get("profiles") or []
            if isinstance(profiles, str):
                profiles = [profiles]

            rule = Rule(
                name=str(item.get("name") or "unnamed"),
                match=str(item.get("match") or ".*"),
                tool=str(item.get("tool") or cfg_default_tool),
                args=[str(a) for a in item.get("args") or []],
                when_minsize=when_minsize,
                when_maxsize=when_maxsize,
                when_mime=item.get("when_mime"),
                profiles=profiles,
            )
            rules.append(rule)

        return cls(rules=rules, default_tool=cfg_default_tool, default_args=cfg_default_args)


def find_config_file(explicit: Optional[str]) -> Optional[str]:
    # If user explicitly requested a path, only return it if it exists.
    if explicit:
        return explicit if os.path.isfile(explicit) else None

    env = os.environ.get("DL_CONFIG")
    if env and os.path.isfile(env):
        return env

    xdg = os.environ.get("XDG_CONFIG_HOME")
    paths = []
    if xdg:
        paths.append(os.path.join(xdg, "dl", "config.toml"))
        paths.append(os.path.join(xdg, "dl", "config.json"))
    home = os.path.expanduser("~")
    paths.append(os.path.join(home, ".config", "dl", "config.toml"))
    paths.append(os.path.join(home, ".config", "dl", "config.json"))
    paths.append("/etc/dl/config.toml")
    paths.append("/etc/dl/config.json")

    for p in paths:
        if os.path.isfile(p):
            return p
    return None


def toml_value(v: Any) -> str:
    """Serialize a Python value into a TOML-ish literal. Good enough for our config."""
    if isinstance(v, str):
        # Use json.dumps for proper escaping; TOML string syntax is compatible enough.
        return json.dumps(v)
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, list):
        return "[ " + ", ".join(toml_value(x) for x in v) + " ]"
    # Fallback: try JSON string
    return json.dumps(str(v))


def dump_config_toml(cfg: Dict[str, Any]) -> str:
    lines: List[str] = []

    if cfg.get("default_tool") is not None:
        lines.append(f"default_tool = {toml_value(cfg['default_tool'])}")
    if "default_args" in cfg:
        lines.append(f"default_args = {toml_value(cfg['default_args'])}")
    lines.append("")

    for rule in cfg.get("rules", []):
        if not isinstance(rule, dict):
            continue
        lines.append("[[rules]]")
        for key in (
            "name",
            "match",
            "tool",
            "args",
            "when_minsize",
            "when_maxsize",
            "when_mime",
            "profiles",
        ):
            if key in rule and rule[key] is not None:
                lines.append(f"{key} = {toml_value(rule[key])}")
        lines.append("")

    text = "\n".join(lines).strip() + "\n"
    return text


def auto_generate_config(explicit: Optional[str], verbose: int) -> Optional[str]:
    """
    If no config file is found, create one with DEFAULT_CONFIG.

    If explicit is given, use that path (adding an extension if missing).
    Otherwise, create under $XDG_CONFIG_HOME or ~/.config/dl/.
    """
    if explicit:
        target = explicit
    else:
        cfg_root = os.environ.get("XDG_CONFIG_HOME")
        if not cfg_root:
            cfg_root = os.path.join(os.path.expanduser("~"), ".config")
        target_dir = os.path.join(cfg_root, "dl")
        # Prefer TOML if tomllib is available, else JSON.
        if tomllib is not None:
            target = os.path.join(target_dir, "config.toml")
        else:
            target = os.path.join(target_dir, "config.json")

    # If no extension provided for explicit path, guess one.
    base, ext = os.path.splitext(target)
    if not ext:
        ext = ".toml" if tomllib is not None else ".json"
        target = base + ext

    try:
        os.makedirs(os.path.dirname(target), exist_ok=True)
        if ext.lower() == ".toml":
            text = dump_config_toml(DEFAULT_CONFIG)
            with open(target, "w", encoding="utf-8") as f:
                f.write(text)
        else:
            with open(target, "w", encoding="utf-8") as f:
                json.dump(DEFAULT_CONFIG, f, indent=2)
                f.write("\n")

        debug_print(
            verbose,
            0,
            f"No config file found; wrote default config to {target!r}.",
        )
        return target
    except Exception as e:
        debug_print(verbose, 0, f"Failed to write default config to {target!r}: {e!r}")
        return None


def load_config(path: Optional[str], verbose: int) -> Tuple[Dict[str, Any], Optional[str]]:
    cfg_path = find_config_file(path)

    if cfg_path is None:
        # Nothing found anywhere, auto-generate user config.
        cfg_path = auto_generate_config(path, verbose)
        if cfg_path is None:
            debug_print(
                verbose,
                0,
                "Using built-in defaults; could not create a config file.",
            )
            return dict(DEFAULT_CONFIG), None

    debug_print(verbose, 1, f"Using config file: {cfg_path}")
    ext = os.path.splitext(cfg_path)[1].lower()
    try:
        with open(cfg_path, "rb") as f:
            if ext == ".toml":
                if tomllib is None:
                    debug_print(
                        verbose,
                        0,
                        "Config file is TOML but tomllib not available. "
                        "Use Python 3.11+ or change to JSON; falling back to defaults.",
                    )
                    return dict(DEFAULT_CONFIG), cfg_path
                data = tomllib.load(f)  # type: ignore[arg-type]
            else:
                # assume JSON
                data = json.load(f)
    except Exception as exc:
        debug_print(verbose, 0, f"Failed to load config '{cfg_path}': {exc!r}")
        return dict(DEFAULT_CONFIG), cfg_path

    # Merge with defaults: user values override
    merged = dict(DEFAULT_CONFIG)
    merged.update({k: v for k, v in data.items() if k in ("default_tool", "default_args", "rules")})
    return merged, cfg_path


def probe_url(
    url: str,
    timeout: float,
    verbose: int,
    user_agent: str,
) -> Optional[ProbeResult]:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return None

    headers = {
        "User-Agent": user_agent or DEFAULT_USER_AGENT,
        "Accept": "*/*",
    }
    req = urllib.request.Request(url, method="HEAD", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            ct = resp.headers.get("Content-Type", "")
            if ";" in ct:
                ct = ct.split(";", 1)[0]
            ct = ct.strip() or None
            cl = resp.headers.get("Content-Length")
            size = int(cl) if cl is not None and cl.isdigit() else None
            debug_print(
                verbose,
                2,
                f"Probe {url!r}: size={human_size(size)}, mime={ct or 'unknown'}",
            )
            return ProbeResult(size=size, mime=ct)
    except urllib.error.HTTPError as e:
        debug_print(verbose, 1, f"HEAD probe failed for {url!r}: HTTP {e.code}")
    except urllib.error.URLError as e:
        debug_print(verbose, 1, f"HEAD probe failed for {url!r}: {e.reason}")
    except Exception as e:
        debug_print(verbose, 1, f"HEAD probe failed for {url!r}: {e!r}")
    return None


def select_rule(
    url: str,
    rules: List[Rule],
    default_tool: str,
    default_args: List[str],
    profile: str,
    probe: Optional[ProbeResult],
    forced_tool: Optional[str],
    verbose: int,
) -> Tuple[str, List[str], Rule]:
    # Forced tool bypasses rule choice but still uses first matching rule for args if any.
    if forced_tool:
        for rule in rules:
            if rule.matches_profile(profile) and rule.matches_url(url):
                debug_print(
                    verbose,
                    1,
                    f"Forced tool {forced_tool!r} but keeping args from rule {rule.name!r}.",
                )
                return forced_tool, list(rule.args), rule
        debug_print(
            verbose,
            1,
            f"Forced tool {forced_tool!r} with no matching rule, using default args.",
        )
        dummy_rule = Rule(
            name="forced",
            match=".*",
            tool=forced_tool,
            args=list(default_args),
        )
        return forced_tool, list(default_args), dummy_rule

    for rule in rules:
        if not rule.matches_profile(profile):
            continue
        if not rule.matches_url(url):
            continue
        if not rule.matches_probe(probe):
            continue
        debug_print(verbose, 1, f"Matched rule {rule.name!r} for URL {url!r}.")
        return rule.tool, list(rule.args), rule

    debug_print(
        verbose,
        1,
        f"No rule matched for URL {url!r}, using default tool {default_tool!r}.",
    )
    fallback_rule = Rule(
        name="__fallback__",
        match=".*",
        tool=default_tool,
        args=list(default_args),
    )
    return default_tool, list(default_args), fallback_rule


def python_http_download(
    url: str,
    output_dir: str,
    verbose: int,
    timeout: float,
    chunk_size: int,
    retries: int,
) -> int:
    parsed = urllib.parse.urlparse(url)
    filename = os.path.basename(parsed.path) or "download"
    dest_path = os.path.join(output_dir or ".", filename)

    headers = {
        "User-Agent": DEFAULT_USER_AGENT,
        "Accept": "*/*",
    }
    attempts = max(1, int(retries) + 1)
    chunk_size = max(1024, int(chunk_size))
    last_error: Optional[Exception] = None

    for attempt in range(1, attempts + 1):
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                cl = resp.headers.get("Content-Length")
                total = int(cl) if cl and cl.isdigit() else None
                debug_print(
                    verbose,
                    1,
                    f"Downloading {url!r} to {dest_path!r} "
                    f"({human_size(total)}).",
                )
                os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
                with open(dest_path, "wb") as f:
                    downloaded = 0
                    last_report = time.time()
                    while True:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        now = time.time()
                        if verbose >= 2 and (now - last_report) > 0.5:
                            if total:
                                pct = downloaded * 100.0 / total
                                debug_print(
                                    verbose,
                                    2,
                                    f"{filename}: {human_size(downloaded)} of {human_size(total)} ({pct:.1f}%)",
                                )
                            else:
                                debug_print(
                                    verbose,
                                    2,
                                    f"{filename}: {human_size(downloaded)} downloaded",
                                )
                            last_report = now
            debug_print(verbose, 1, f"Finished {dest_path!r}.")
            return 0
        except Exception as e:
            last_error = e
            if attempt >= attempts:
                break
            log_warning(f"Download attempt {attempt} for {url!r} failed: {e!r} -- retrying")
            time.sleep(min(2.0 * attempt, 5.0))

    log_error(f"Python HTTP download failed for {url!r}: {last_error!r}")
    return 1


def apply_tool_defaults(tool: str, args: List[str]) -> List[str]:
    """Ensure sane defaults when a rule provided no args."""
    if tool == "curl" and not args:
        # Avoid dumping response bodies to stdout while still following redirects.
        return ["-L", "-O"]
    return args


def run_external_command(
    tool: str,
    tool_args: List[str],
    url: str,
    extra_tool_args: List[str],
    dry_run: bool,
    verbose: int,
) -> int:
    cmd = [tool] + tool_args + extra_tool_args + [url]
    debug_print(verbose, 1, "Command: " + " ".join(map(shlex_quote, cmd)))
    if dry_run:
        return 0

    try:
        proc = subprocess.run(cmd)
        return proc.returncode
    except FileNotFoundError:
        log_error(f"Tool {tool!r} not found in PATH.")
        show_install_instructions(tool)
        return 127
    except KeyboardInterrupt:
        log_warning("Interrupted by user.")
        return 130
    except Exception as e:
        log_error(f"Failed to run {tool!r}: {e!r}")
        return 1


def shlex_quote(s: str) -> str:
    # Minimal shell quoting for debug output
    if not s:
        return "''"
    if re.search(r"[^\w@%+=:,./-]", s):
        return "'" + s.replace("'", "'\"'\"'") + "'"
    return s


def gather_urls(args: argparse.Namespace) -> List[str]:
    urls: List[str] = []
    if args.urls:
        urls.extend(args.urls)

    if args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        urls.append(line)
        except Exception as e:
            log_error(f"Failed to read URL file {args.file!r}: {e!r}")

    if not urls and not sys.stdin.isatty():
        for line in sys.stdin:
            line = line.strip()
            if line:
                urls.append(line)

    return urls


def print_effective_config(config: Dict[str, Any]) -> None:
    text = json.dumps(config, indent=2, sort_keys=True)
    sys.stdout.write(text + "\n")


def iter_rule_dicts(config: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for rule in config.get("rules") or []:
        if isinstance(rule, dict):
            yield rule


def list_tools(raw_config: Dict[str, Any], effective: EffectiveConfig) -> None:
    tools = {effective.default_tool}
    for rule in iter_rule_dicts(raw_config):
        tool = rule.get("tool")
        if tool:
            tools.add(str(tool))
    sys.stdout.write("Configured tools:\n")
    for tool in sorted(tools):
        sys.stdout.write(f"  - {tool}\n")


def list_profiles(raw_config: Dict[str, Any]) -> None:
    profiles = set()
    for rule in iter_rule_dicts(raw_config):
        entry = rule.get("profiles")
        if not entry:
            profiles.add("<all>")
            continue
        if isinstance(entry, str):
            profiles.add(entry)
            continue
        for item in entry:
            profiles.add(str(item))
    sys.stdout.write("Known profiles:\n")
    for profile in sorted(profiles):
        sys.stdout.write(f"  - {profile}\n")


def list_rules(raw_config: Dict[str, Any]) -> None:
    sys.stdout.write("Rules:\n")
    for rule in iter_rule_dicts(raw_config):
        name = str(rule.get("name") or "unnamed")
        match = str(rule.get("match") or "*")
        tool = str(rule.get("tool") or "<default>")
        profiles = rule.get("profiles") or []
        if isinstance(profiles, str):
            profiles = [profiles]
        constraints: List[str] = []
        if rule.get("when_minsize"):
            constraints.append(f"minsize={rule['when_minsize']}")
        if rule.get("when_maxsize"):
            constraints.append(f"maxsize={rule['when_maxsize']}")
        if rule.get("when_mime"):
            constraints.append(f"mime={rule['when_mime']}")
        constraint_text = ", ".join(constraints) if constraints else "none"
        profile_text = ", ".join(str(p) for p in profiles) if profiles else "all"
        sys.stdout.write(
            f"  - {name}: tool={tool}, profiles={profile_text}, match={match}, constraints={constraint_text}\n"
        )


def list_installers() -> None:
    installers = detect_installers()
    if not installers:
        sys.stdout.write("No helper installers detected; falling back to full hint matrix.\n")
        return
    sys.stdout.write("Detected installers:\n")
    for installer in installers:
        sys.stdout.write(f"  - {installer}\n")


def explain_selection(
    url: str,
    tool: str,
    tool_args: List[str],
    extra_tool_args: List[str],
    rule: Rule,
    probe: Optional[ProbeResult],
) -> None:
    probe_text = "size=unknown, mime=unknown"
    if probe:
        size = human_size(probe.size)
        mime = probe.mime or "unknown"
        probe_text = f"size={size}, mime={mime}"
    args_preview = " ".join(map(shlex_quote, tool_args)) or "<none>"
    extra_preview = " ".join(map(shlex_quote, extra_tool_args)) or "<none>"
    full_cmd = " ".join(
        map(shlex_quote, [tool] + tool_args + extra_tool_args + [url])
    )
    sys.stdout.write(
        f"URL: {url}\n"
        f"  Rule: {rule.name} ({rule.match})\n"
        f"  Tool: {tool}\n"
        f"  Rule args: {args_preview}\n"
        f"  Extra args: {extra_preview}\n"
        f"  Probe: {probe_text}\n"
        f"  Command: {full_cmd}\n"
    )


def main(argv: Optional[List[str]] = None) -> int:
    env_profile = os.environ.get("DL_PROFILE") or "default"
    env_tool = os.environ.get("DL_TOOL")
    env_tool_args_raw = os.environ.get("DL_TOOL_ARGS") or ""
    env_tool_args = shlex.split(env_tool_args_raw) if env_tool_args_raw.strip() else []
    env_log_format = (os.environ.get("DL_LOG_FORMAT") or "plain").strip().lower()
    if env_log_format not in {"plain", "json"}:
        env_log_format = "plain"
    env_log_file = os.environ.get("DL_LOG_FILE")
    if env_log_file:
        env_log_file = env_log_file.strip() or None
    env_log_timestamps = env_flag("DL_LOG_TIMESTAMPS")
    env_log_theme = (os.environ.get("DL_LOG_THEME") or "default").strip().lower()
    if env_log_theme not in LOG_THEMES:
        env_log_theme = "default"
    env_log_color = (os.environ.get("DL_LOG_COLOR") or "auto").strip().lower()
    if env_log_color not in {"auto", "always", "never"}:
        env_log_color = "auto"

    parser = argparse.ArgumentParser(
        prog="dl",
        description="Smart download dispatcher that chooses tools like curl, yt-dlp, aria2c, etc.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Examples:
              dl https://example.com/file.iso
              dl -v https://youtu.be/dQw4w9WgXcQ
              dl -P audio https://youtu.be/dQw4w9WgXcQ
              dl -n -T curl https://example.com/file.iso
              echo "https://example.com/1.iso" | dl

            Configuration:
              On first run, dl will auto-generate a default config in:
                $XDG_CONFIG_HOME/dl/config.toml (or config.json)
              or:
                ~/.config/dl/config.toml (or config.json)

              You can override the path with:
                --config /path/to/config
              or:
                DL_CONFIG=/path/to/config dl ...
            """
        ),
    )

    parser.add_argument(
        "urls",
        nargs="*",
        help="URLs to download. If omitted, read from stdin.",
    )
    parser.add_argument(
        "-f",
        "--file",
        help="Read URLs (one per line) from a file.",
    )
    parser.add_argument(
        "-T",
        "--tool",
        default=env_tool,
        help="Force a specific tool (curl, wget, yt-dlp, aria2c, gallery-dl, python-http, ...).",
    )
    parser.add_argument(
        "-P",
        "--profile",
        default=env_profile,
        help="Profile name used to filter rules (default: %(default)s).",
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Do not execute, just show what would be run.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (can be used multiple times).",
    )
    parser.add_argument(
        "--log-format",
        choices=["plain", "json"],
        default=env_log_format,
        help="Logging format for stderr output (default: %(default)s).",
    )
    parser.add_argument(
        "--log-file",
        default=env_log_file,
        help="Append logs to this file in addition to stderr.",
    )
    parser.add_argument(
        "--log-timestamps",
        dest="log_timestamps",
        action="store_true",
        default=None,
        help="Include ISO 8601 timestamps in log lines.",
    )
    parser.add_argument(
        "--no-log-timestamps",
        dest="log_timestamps",
        action="store_false",
        help="Disable timestamps in log lines.",
    )
    parser.add_argument(
        "--log-theme",
        choices=sorted(LOG_THEMES.keys()),
        default=env_log_theme,
        help="Color theme for log levels (default: %(default)s).",
    )
    parser.add_argument(
        "--log-color",
        choices=["auto", "always", "never"],
        default=env_log_color,
        help="Control ANSI color output for logs (default: %(default)s).",
    )
    parser.add_argument(
        "--config",
        help="Path to a TOML or JSON config file.",
    )
    parser.add_argument(
        "--no-probe",
        action="store_true",
        help="Skip HTTP HEAD probing for size and MIME type.",
    )
    parser.add_argument(
        "--probe-timeout",
        type=float,
        default=5.0,
        help="Timeout in seconds for HTTP HEAD probes (default: %(default)s).",
    )
    parser.add_argument(
        "--probe-user-agent",
        default=DEFAULT_USER_AGENT,
        help="Custom User-Agent header for HTTP HEAD probes.",
    )
    parser.add_argument(
        "-A",
        "--tool-arg",
        action="append",
        default=[],
        help="Additional argument to pass to the chosen tool (can be repeated).",
    )
    parser.add_argument(
        "-C",
        "--output-dir",
        default=".",
        help="Output directory for python-http builtin downloader (default: current directory).",
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Load and print the effective config as JSON, then exit.",
    )
    parser.add_argument(
        "--show-config-path",
        action="store_true",
        help="Print the resolved config path and exit.",
    )
    parser.add_argument(
        "--list-tools",
        action="store_true",
        help="List all tools referenced by the current config and exit.",
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="List all profiles referenced by the current config and exit.",
    )
    parser.add_argument(
        "--list-rules",
        action="store_true",
        help="List configured rules with summary info and exit.",
    )
    parser.add_argument(
        "--list-installers",
        action="store_true",
        help="List detected package installers for helper binaries and exit.",
    )
    parser.add_argument(
        "--no-install-hints",
        action="store_true",
        help="Suppress install hints for missing helper tools.",
    )
    parser.add_argument(
        "--download-timeout",
        type=float,
        default=30.0,
        help="Timeout in seconds for the builtin python-http downloader.",
    )
    parser.add_argument(
        "--download-retries",
        type=int,
        default=2,
        help="Number of retries for the builtin python-http downloader.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=64 * 1024,
        help="Chunk size in bytes for the builtin python-http downloader.",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Explain which rule/tool would run for each URL and imply --dry-run.",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print version info and exit.",
    )

    ns = parser.parse_args(argv)

    if ns.log_timestamps is None:
        ns.log_timestamps = env_log_timestamps
    ns.log_file = ns.log_file or None
    if ns.log_file:
        ns.log_file = os.path.expanduser(ns.log_file)
    use_color = should_colorize(ns.log_color, ns.log_format)
    configure_logging(
        verbose=ns.verbose,
        fmt=ns.log_format,
        timestamps=bool(ns.log_timestamps),
        log_path=ns.log_file,
        color=use_color,
        theme_name=ns.log_theme,
    )

    if env_tool_args:
        ns.tool_arg = list(env_tool_args) + ns.tool_arg

    if ns.no_install_hints:
        suppress_install_hints()

    if ns.explain:
        ns.dry_run = True

    if ns.version:
        sys.stdout.write(f"dl {__version__} (Python dispatcher with auto-config)\n")
        return 0

    raw_config, cfg_path = load_config(ns.config, ns.verbose)
    effective_config = EffectiveConfig.from_dict(raw_config)

    info_action = False
    if ns.show_config:
        print_effective_config(raw_config)
        info_action = True
    if ns.show_config_path:
        if cfg_path:
            sys.stdout.write(f"{cfg_path}\n")
        else:
            sys.stdout.write("No config file found; using built-in defaults.\n")
        info_action = True
    if ns.list_tools:
        list_tools(raw_config, effective_config)
        info_action = True
    if ns.list_profiles:
        list_profiles(raw_config)
        info_action = True
    if ns.list_rules:
        list_rules(raw_config)
        info_action = True
    if ns.list_installers:
        list_installers()
        info_action = True
    if info_action:
        return 0

    urls = gather_urls(ns)
    if not urls:
        parser.error("No URLs supplied via args, file, or stdin.")

    # Main loop
    overall_rc = 0
    for url in urls:
        url = url.strip()
        if not url:
            continue

        probe = None
        if not ns.no_probe:
            probe = probe_url(
                url=url,
                timeout=ns.probe_timeout,
                verbose=ns.verbose,
                user_agent=ns.probe_user_agent,
            )

        tool, tool_args, rule = select_rule(
            url=url,
            rules=effective_config.rules,
            default_tool=effective_config.default_tool,
            default_args=effective_config.default_args,
            profile=ns.profile,
            probe=probe,
            forced_tool=ns.tool,
            verbose=ns.verbose,
        )
        tool_args = apply_tool_defaults(tool, tool_args)

        if ns.explain:
            explain_selection(url, tool, tool_args, ns.tool_arg, rule, probe)

        if tool == "python-http":
            debug_print(
                ns.verbose,
                1,
                f"Using builtin Python HTTP downloader for {url!r} (rule {rule.name!r}).",
            )
            if ns.dry_run:
                continue
            rc = python_http_download(
                url=url,
                output_dir=ns.output_dir,
                verbose=ns.verbose,
                timeout=ns.download_timeout,
                chunk_size=ns.chunk_size,
                retries=ns.download_retries,
            )
        else:
            rc = run_external_command(
                tool=tool,
                tool_args=tool_args,
                url=url,
                extra_tool_args=ns.tool_arg,
                dry_run=ns.dry_run,
                verbose=ns.verbose,
            )

        if rc != 0:
            overall_rc = rc

    return overall_rc


if __name__ == "__main__":
    raise SystemExit(main())
