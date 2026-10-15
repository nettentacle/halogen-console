# Changelog

Notable changes to the HALOGEN Console (8733).

## 2026-10-14 — Halogen checkpoint switcher
- Per-engine checkpoint selection for halogen: pick which `.hgn` file the
  `halogen-docker.service` loads (`v2` / `abliterated`; the registry is one
  line per checkpoint).
- Switching atomically rewrites the `HALOGEN_CHECKPOINT=` line (tmp +
  `os.replace`), keeps a rolling backup, inserts a timestamped comment, and
  runs `daemon-reload`. A named-but-different checkpoint turns even `start`
  into stop→wait→start; responses carry `checkpoint` / `checkpoint_switched`.
- Always `systemctl --user reset-failed` before starting: rapid switches
  otherwise trip the unit's `StartLimitBurst` (`start-limit-hit`).
- Frontend: `#hCheckpoint` select (shown only when halogen is active),
  follows the running instance / unit file like the kyojin variant selector,
  full zh/en labels, confirm dialogs name the checkpoint.

## 2026-10-11 — NPU row for container-held NPU
- When the NPU is used by a root process inside the halogen container, the
  user-level unit can't read its `fdinfo`; a persistent `docker exec` stream
  (reads inside the container, 0.5 s cadence, self-healing) feeds the row.

## 2026-10-08 — Metrics attribution + probe gating + NPU row
- Probe order fixed: kyojin's `/stats` is asked **before** halogen's
  `/metrics` (kyojin 1.3 ships llamacpp-style `/metrics` that used to be
  misattributed to halogen).
- `engine_on_8731()` gate (systemd-state based, 3 s cache): the `/stats`
  probe is only sent when kyojin (or nothing) owns 8731 — ends the
  ~1 req/s 404 flood in halogen's access log; `_PROBE_NOISE` extended to
  `/stats`.
- System-resources section gains an NPU row (amdxdna utilization via
  per-client `fdinfo` engine counters, baseline dropped when clients exit).

## 2026-10-07 — Strata engine, kyojin variants, gsqhalo
- `strata` engine (port 8088): `/health` + `/v1/status`; since it exposes no
  cumulative counters, the panel accumulates its own rate history from
  `last_timings` of newly completed requests (reset on engine restart).
- kyojin **model variant** selector (GLM / Qwen lanes): per-variant units,
  `?variant=` plumbing, stop-without-variant stops all variants, autostart
  enables one variant and disables the rest.
- `gsqhalo` engine (podman container, port 8081) added to the registry.
- Header fits `HALOGEN / KYOJIN / STRATA` + pills + actions at 1280×720.

## 2026-10-05 — Second engine (kyojin)
- kyojin joins the registry; port-8731 sharing handled by enforcing
  exclusivity: start/restart stop the other engine first, `restart` =
  stop → wait for `MemAvailable` to recover → start.
- kyojin's "version" is its git HEAD; `/api/update` compares it against
  upstream via the GitHub compare API (note: `ahead_by`/`behind_by` are
  relative to the *head* and must stay swapped).
- Frontend engine pills + `localStorage` selection; the page auto-follows
  the running engine on first visit.

## 2026-10-01 — UI redesign + Docker adaptation
- CSS consolidated from three stacked generations into one layer
  ("仪表台" design language: mono readouts, single glowing status dot,
  accent reserved for state/peak).
- halogen moved to Docker: unit `halogen-docker`, absolute `/snap/bin/docker`,
  all log reads switched to `docker logs` (podman-era dedup removed).
- aiohttp dependency moved to the apt package `python3-aiohttp` (survives
  OS python upgrades; a user-site pip copy would shadow it).

## 2026-09-27 — Bilingual UI + upstream version check
- zh/en i18n: `data-i18n` tags + `I18N` dict, `applyLang()` re-renders from
  in-memory state, choice persisted in `localStorage` (default zh); server
  error strings follow the `X-Lang` header.
- `/api/update`: compares the engine's reported version against the newest
  GitHub tag (tags API — the repo publishes tags but no releases; numeric
  tuple compare; 300 s cache / 60 s on failure).
