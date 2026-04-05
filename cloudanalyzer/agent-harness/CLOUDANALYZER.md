# CloudAnalyzer CLI Harness — Architecture

## Overview

CloudAnalyzer is a **CLI-first** Python package for point cloud QA. Unlike most
CLI-Anything harnesses that bridge a GUI application to the command line,
this harness wraps an existing CLI tool to provide:

1. Standardized Click interface with `--json` on every command
2. Project/session state management with undo/redo
3. SKILL.md for agent discovery
4. REPL mode

## Backend Strategy

Since CloudAnalyzer is a Python package, the backend (`ca_backend.py`) imports
CloudAnalyzer functions directly — no subprocess invocation needed. This makes
the harness faster and more reliable than subprocess-based approaches.

## Command Mapping

| Harness Group | CloudAnalyzer Command(s) |
|---|---|
| evaluate run | ca evaluate |
| evaluate compare | ca compare |
| evaluate diff | ca diff |
| evaluate ground | ca ground-evaluate |
| trajectory evaluate | ca traj-evaluate |
| trajectory batch | ca traj-batch |
| check run | ca check |
| check init | ca init-check |
| baseline decision | ca baseline-decision |
| baseline save | ca baseline-save |
| baseline list | ca baseline-list |
| process downsample | ca downsample |
| process split | ca split |
| info show | ca info |

## State Model

The project JSON tracks:
- Loaded cloud and trajectory file paths
- QA results from evaluation commands
- Operation history with timestamps
- Session settings

Operations are recorded automatically and support undo/redo via
the Session class.
