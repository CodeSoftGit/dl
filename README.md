dl
==

`dl` is a smart download dispatcher that routes URLs to the best tool. It can run
`yt-dlp`, `gallery-dl`, `aria2c`, `curl`, `wget`, or its own minimal HTTP fetcher
based on URL patterns, file sizes, and MIME types defined in a simple config file.

Features
--------
- Auto-detects the best download backend via pattern based rules.
- Loads configuration from TOML/JSON, auto-generating a default template on first run.
- Prints verbose logging, supports dry-runs, forced tools, per-profile rule sets, and
  additional custom arguments.
- Ships as a standard Python package (PEP 621) published as `open-dl-py`, so `pip install open-dl-py`
  or `pipx install open-dl-py` yields a `dl` console script.
- Emits actionable install hints for missing helper tools, covering pip/pipx,
  Homebrew, apt/.deb, and dnf/.rpm flows.

Installation
------------
The project follows a standard `src/` layout and exposes a `dl` console script.
Common installation paths:

### pip
```bash
python3 -m pip install open-dl-py
```

### pipx
```bash
pipx install open-dl-py
```

### From source checkout
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Helper Tool Installation Matrix
-------------------------------
`dl` depends on external CLIs. If one is missing, the dispatcher prints install
instructions for every detected package ecosystem (pip/pipx, Homebrew, apt/.deb,
dnf/.rpm). The table below mirrors the built-in hints.

| Tool        | pip                              | pipx                        | Homebrew                | Debian/Ubuntu (`apt`) | Fedora/RHEL (`dnf`) |
|-------------|----------------------------------|-----------------------------|-------------------------|-----------------------|---------------------|
| `yt-dlp`    | `python3 -m pip install yt-dlp`  | `pipx install yt-dlp`       | `brew install yt-dlp`   | `sudo apt install yt-dlp` | `sudo dnf install yt-dlp` |
| `gallery-dl`| `python3 -m pip install gallery-dl` | `pipx install gallery-dl` | `brew install gallery-dl` | *(use pip/pipx)*      | *(use pip/pipx)*    |
| `aria2c`    | *(system package)*               | *(system package)*          | `brew install aria2`    | `sudo apt install aria2` | `sudo dnf install aria2` |
| `curl`      | *(bundled with OS)*              | *(bundled with OS)*         | `brew install curl`     | `sudo apt install curl`  | `sudo dnf install curl`  |
| `wget`      | *(system package)*               | *(system package)*          | `brew install wget`     | `sudo apt install wget`  | `sudo dnf install wget`  |

When no manager is detected locally, the hints fallback to the full matrix to make
it easy to copy/paste commands for any environment.
For downloaded package files the same commands work (`sudo apt install ./pkg.deb`
for `.deb`, `sudo dnf install ./pkg.rpm` for `.rpm`).

Usage
-----
The CLI exposes a number of switches:

```bash
# Download a single URL
dl https://example.com/big.iso

# Force a tool and inspect what would run
dl -n -T curl https://example.com/file

# Use the audio profile with verbose logging
dl -P audio -vv https://youtu.be/dQw4w9WgXcQ

# Read URLs from stdin
cat urls.txt | dl --probe-timeout 2

# Inspect what dl would do for a URL without running it
dl --explain https://example.com/file.iso

# Browse configured tools, profiles, and rules
dl --list-tools
dl --list-profiles
dl --list-rules
dl --list-installers

# Emit JSON logs with timestamps and also tee them to a file
dl --log-format json --log-timestamps --log-file ~/dl.log https://example.com/file

# Use the vibrant theme with forced colors (even when piped)
dl --log-theme vibrant --log-color always https://example.com/file
```

When `dl` falls back to plain `curl` (no rule args configured), it now injects `-L -O`
automatically so responses are saved to files rather than dumped to your terminal.

Configuration
-------------
`dl` keeps all routing logic in a single TOML or JSON config file. On first run the
dispatcher auto-generates `~/.config/dl/config.toml` (or `config.json` when TOML
support is missing). Override the location with `--config /path/to/config` or by
setting `DL_CONFIG`.

The top-level keys are:

- `default_tool`: fallback command when no rule matches (`curl`, `wget`, `python-http`, etc).
- `default_args`: extra arguments appended when the default tool kicks in.
- `rules`: an array of rule objects. Each rule understands:
  - `name`: label shown in `--explain` output.
  - `match`: regular expression applied to the URL.
  - `tool`: command name to run when the rule matches.
  - `args`: list of arguments passed to the tool before user supplied `-A/--tool-arg`.
  - `profiles`: optional list of profile names; empty list means “all profiles”.
  - `when_minsize` / `when_maxsize`: optional size guards (accepts `50M`, `1G`, etc).
  - `when_mime`: optional MIME glob (`video/*`, `application/pdf`, …).

A minimal TOML example:

```toml
default_tool = "curl"
default_args = [ "-L", "-O" ]

[[rules]]
name = "ytdlp_video"
match = "youtube\\.com|youtu\\.be"
tool = "yt-dlp"
args = [ "-f", "bv*+ba/b" ]
profiles = [ "default", "video" ]

[[rules]]
name = "audio_profile"
match = "youtube\\.com|youtu\\.be"
tool = "yt-dlp"
args = [ "-x", "--audio-format", "mp3" ]
profiles = [ "audio" ]
```

Use `dl --show-config` to print the effective JSON form (after merging defaults),
and `dl --list-rules` / `--list-tools` / `--list-profiles` to sanity-check your edits.

Introspection & Planning
------------------------
- `dl --show-config-path` prints the resolved config path (auto-generated if needed).
- `dl --show-config` still dumps the full effective configuration as JSON.
- `dl --list-tools`, `--list-profiles`, and `--list-rules` provide quick summaries of how URLs map to tools.
- `dl --list-installers` reports which helper package managers were detected for install hints.
- `dl --explain URL [...]` now implies `--dry-run` and prints the exact rule, tool, arguments, and probe information for each URL.

Logging
-------
You can tune how status output is rendered without editing the code:

- `--log-format {plain,json}` switches between human-friendly and structured logs.
- `--log-timestamps/--no-log-timestamps` toggles ISO 8601 timestamps in log lines.
- `--log-file PATH` mirrors stderr logging to the given file within the same format.
- `--log-theme {default,vibrant}` selects the ANSI palette used when coloring plain logs.
- `--log-color {auto,always,never}` controls whether color is emitted (auto only colors when stderr is a TTY).

Builtin Downloader Controls
---------------------------
When falling back to the builtin python-http downloader you can now fine-tune its behavior:

- `--download-timeout SECONDS` (default `30`) feeds the urllib timeout.
- `--download-retries N` retries failed builtin downloads `N` times (defaults to 2, making 3 attempts total).
- `--chunk-size BYTES` changes the streaming chunk size (default `65536`).
- `--probe-user-agent STRING` customizes the HEAD probe User-Agent, and `--no-install-hints` or `DL_NO_INSTALL_HINTS=1` silences missing tool guidance outright.

Environment Variables
---------------------
`dl` honors a few convenience variables so you can set defaults in your shell profile:

- `DL_PROFILE` sets the default `-P/--profile`.
- `DL_TOOL` sets the default forced tool (`-T/--tool`) without typing the flag.
- `DL_TOOL_ARGS="--foo --bar=baz"` injects extra tool args via shell-style parsing (applied before repeated `-A` flags).
- `DL_NO_INSTALL_HINTS=1` disables all helper tool hints even when a command is missing.
- `DL_LOG_FORMAT=plain|json`, `DL_LOG_TIMESTAMPS=1`, `DL_LOG_THEME=default|vibrant`, `DL_LOG_COLOR=auto|always|never`, and `DL_LOG_FILE=/path/to/file.log` provide defaults for the corresponding CLI switches.

Development
-----------
```bash
python -m pip install -U pip build
python -m build
```

Run the binary from the repository root without installing:

```bash
python main.py --help
```
