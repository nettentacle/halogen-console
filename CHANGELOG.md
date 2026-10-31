# Changelog

Notable changes to the HALOGEN Console (8733).

## 2026-10-31 — glass skin 入库
- 仓库收纳第二套前端皮肤 `skins/glass.html`（即 `~/halogen-control-glass/`
  的玻璃拟态版）：磨砂半透明卡片、鼠标聚光、自绘玻璃下拉（原生 option
  做不了磨砂）、已加载模型行带检查点档位、明暗主题。
- **后端零改动**：玻璃目录里的 `halogen-control.py` 与经典版**逐字节相同**
  ——"皮肤"就是另一份 html；后端按 mtime 热加载，
  `cp skins/glass.html halogen-control.html` + 浏览器刷新即完成切换。
- `docs/` 新增明暗两张玻璃版实况截图（`screenshot-glass-{dark,light}.png`）。

## 2026-10-29 — 跟随引擎 API 鉴权（HALOGEN_API_KEY）
- halogen 开启本地 API 鉴权后（除 `/health` 外全部路由要
  `Authorization: Bearer <key>`），面板新增 `_auth_headers()` + `_get()`：
  读与容器挂载的**同一份宿主密钥文件**（600），对本机引擎的所有请求统一带
  鉴权头——六个引擎都带、只有 halogen 校验，省得按"这会儿谁在答话"分支。
  密钥文件读不到就返回空头（等于没开鉴权，照旧工作）。
- 对外部主机（GitHub tags / compare）的请求**显式传 `headers=`**，
  不带本地引擎的密钥——密钥不出机器。同日引入时的一个形参缺陷见下条。

## 2026-10-29 — 修复：更新检查全部失败
- 引入本地 API 鉴权时新增的 `_get()` 助手只接受 `(session, url)`，内部固定附加
  引擎鉴权头；而三处 GitHub 查询（tags / compare）以 `headers=` 关键字调用它，
  全部抛 `TypeError: _get() got an unexpected keyword argument 'headers'`，
  六引擎的 `/api/update` 一律报错。
- `_get()` 现在接受可选 `headers`：显式传入（外部请求）时原样使用、不带本地
  引擎的 `Authorization` 头；未传时行为不变。

## 2026-10-22 — kyojin 收为 Qwen 单车道
- GLM-5.3-Flash 车道与权重回收（主机清盘），kyojin 只剩 `kyojin-qwen`
  一个 systemd 单元：`ENGINES["kyojin"]` 的 `variants` 注册表摘除
  （`unit`/`model`/`log_unit` 直接指向 Qwen 车道），`_UNITS_8731` 去掉
  `kyojin-glm`。
- 前端引擎注册表同步：kyojin 不再渲染模型子选择器（变体机制本身保留，
  引擎登记了 `variants` 就会出现）；默认变体常量落到 `qwen`。
- 服务状态/视觉行照旧来自 `/v1/models` 与 `/health`（Qwen 车道行为不变）。

## 2026-10-21 — gufo（第六个引擎）
- Added `gufo` to `ENGINES`: unit `gufo`, port 8731, rootless podman container
  (`ghcr.io/gufo-org/toolboxes/gufo-runtime`), Qwen3.8-Flash-Next UD-Q4_K_XL +
  MTP draft + vision, 2 sessions at native 262144 context. Its HTTP surface is
  deliberately llama.cpp-shaped (`/health`, `/slots`, `/props`, `/metrics`), so
  the panel reuses the gsqhalo parsers; `/props.model_alias` is the
  gufo-specific identity marker and `build_info` (`0.20.0 (0c03748)`) supplies
  the version for the tag-based update check against `gufo-org/gufo`.
- `engine_state_gufo`: readiness from `/health`, context/slots/busy from
  `/slots`, vision from `/props.modalities.vision`.
- `_metrics_gufo`: llama.cpp Prometheus counters (`llamacpp:*`) plus gufo's
  `kv_cache_usage_ratio`; same counter-delta rate logic as gsqhalo.
- Headroom cell: shows the `host_available_mib` reported by gufo's startup
  `load_completed` line (labelled "host free after load").
- Fixed: the engine card's version was hidden for engines whose `/metrics` has
  no `requests_total` (true for gufo) — it now shows any self-reported version.
- Fixed: `/api/update` responses could race when the page first auto-follows the
  running engine, leaving the previous engine's upstream version on the card.

## 2026-10-18 — 引擎无响应不再显示成"正在加载模型"
- 背景：内核内存压缩（compact_stall）会把 halogen 引擎的 PING 线程卡住
  45–170 s，期间 `/health` 超时或返回 503；面板把"未就绪"一律显示为
  「正在加载模型…」，看起来像引擎在反复重载（实际没有重启，日志里只有一次
  `model ready`）。
- **Backend**: halogen 的 `/health` 超时/断连时，只有"本次运行从未就绪过"
  才报 `unreachable`（首启加载中）。一旦本次运行就绪过（按 systemd
  `ActiveEnterTimestampMonotonic` 判定），后续问不到就报
  `engine_unresponsive`——区分加载与卡死。`/health` 自己返回 503 时携带的
  `engine_unresponsive` 原样透传。
- **Frontend**: 顶栏对 `engine_unresponsive` 显示「引擎无响应（内核内存压缩）」
  并改用红点；悬停提示说明压缩结束会自动恢复、不是重新加载。详情行
  「模型就绪」显示「无响应（内存压缩）」而不是原始状态串。首启加载仍显示
  「正在加载模型…」。

## 2026-10-17 — hipfire metrics from /metrics
- hipfire's KPI/timeline now reads `/metrics` (Prometheus) instead of `/stats`.
  It has `hipfire_requests_total`, prompt/completion token counters, and
  per-request `hipfire_prefill_tokens_per_second` /
  `hipfire_decode_tokens_per_second` histograms.
- Counter deltas fill the completed-request table's prompt tokens, output
  tokens, prefill tok/s and decode tok/s; histogram `sum/count` gives the
  averages. **Peaks are the maximum actual sampled per-request value since
  panel start** (histogram buckets are not used; peaks reset on panel restart).
- Cache hit is `cached_prompt_tokens / prompt_tokens`. A true draft
  acceptance rate is not exposed; the panel shows hipfire's cumulative mean
  accepted length τ (`spec_tau`) instead, labelled "草稿 τ（累计接受长度）"
  (per-request τ stays in the engine log). Page-cache headroom is a
  halogen-specific contiguous-block line; for hipfire the panel shows the
  startup `free_after_weights` as "权重后可用" (91.8 GiB on the live box).
  KV usage remains `—`.
- Verified on the live 27B: 28 prompt / 58 output tokens produced a table row
  with prefill 113.5 and decode 94.1; KPI averages 295.4 / 32.8 and peaks
  113.5 / 94.1 (the peaks equal that sampled request).

## 2026-10-17 — all engines on port 8731
- `strata` now runs `--port 8731` (was 8088), `gsqhalo` maps
  `-p 8731:8731` and listens on `--port 8731` (was 8081), and hipfire is pinned
  to `serve.port = 8731` in `~/.hipfire/config.toml` (was 11435). halogen and
  kyojin were already on 8731.
- The panel's `ENGINES` endpoints all point at `127.0.0.1:8731`;
  `_UNITS_8731` now covers all five units (including kyojin's two variants) and
  `engine_metrics()` dispatches by systemd active unit instead of probing every
  port. `engine_state_*` gained engine-specific marker guards (`version` dict,
  `/slots`, `/v1/status` `engine`, `/props`, `/stats` `requests_served`) so a
  non-active engine can't be mistaken for the one that owns 8731.
- Verified sequentially through the panel: hipfire → strata → gsqhalo →
  halogen, all bound 8731, each with the correct `metrics.engine` attribution;
  halogen 0.18.7 restored at the end. External scripts that still use
  8088/8081/11435 must be updated.

## 2026-10-17 — hipfire: single slot at 262144 (2-slot reverted)
- Removed `HIPFIRE_SERVE_MULTI_SLOT*` from `hipfire-27b.service`; the unit keeps
  `--max-seq 262144` with DFlash and vision. Single-slot uses VMM q8 KV and
  runs at **126.7 t/s** on the official merge-sort code prompt (3 runs,
  τ 13.18; 130.4 at 32768, within run-to-run/power noise).
- GTT drops from ~41 GB (2 slots) to ~21.8 GiB; panel card shows
  `262144 / 1`; `min_avail_gib` is back to 40.
- The 2-slot experiment was verified (2 × 262144, concurrent requests and
  vision) but measured ~32 t/s on a 256-token code prompt: legacy KV,
  `max_batch_tokens` clamped to 2048, continuous batching deferred. Reverted
  in favour of the single-slot fast path.

## 2026-10-17 — hipfire 27B (fifth engine, vision)
- Added `hipfire` to `ENGINES`: unit `hipfire-27b`, port 11435, `/health` +
  `/stats`, journal logs. The unit runs Qwen3.8-27B MQ4 XTS with DFlash and
  `--vision` + `HIPFIRE_VISION_MODE=on` (vision tower enabled).
- `engine_state_hipfire`: readiness from `/health` (a non-null `loading_model`
  means loading), version from the checkout's exact tag (`v0.4.2.1`), context
  and slots from `/health`, queue/requests from `/stats`; vision comes from the
  unit configuration because `/health` does not expose it.
- `_metrics_hipfire`: recent-window decode rate, queue depth and request count;
  appends a trend point when `requests_served` grows; cumulative averages,
  draft/cache/KV metrics render `—`.
- `h_action` honors an engine-level `min_avail_gib`; hipfire starts at
  55 GiB free (2 × 262144 slots + DFlash + vision is about 41 GB GTT).
- The hipfire unit runs the experimental multi-slot engine with 2 slots at
  262144 context each (`HIPFIRE_SERVE_MULTI_SLOT=1`, `..._SLOTS=2`,
  `..._CTX=262144`, `--max-seq 262144`). This forces legacy KV storage and
  clamps `max_batch_tokens` to 2048 (2 × 1024 scratch); the single-slot
  benchmark numbers do not apply to this route. Two concurrent requests and
  a vision request were verified.
- `h_update` compares hipfire against `warpfront/hipfire` tags, filtering
  non-version tags (`safety/win_*`); the update tooltip is hipfire-specific.
- Frontend: fifth pill; five-engine i18n/confirm text; compacted header
  padding/gaps so five pills + checkpoint select + three action buttons + EN/
  theme still fit one row at 1280×720; hipfire has no variant/checkpoint
  selectors.
- Verified end to end: panel stop/start, cross-engine switch
  (`stopped_others:["hipfire"]`), two concurrent requests on the 2-slot
  engine, vision red-square → `Red`, and 1280×720 screenshots of both running
  and stopped states. Single-slot DFlash measured 130.4 t/s code; the
  experimental 2-slot route measured ~32 t/s on a 256-token code prompt
  (legacy KV, `max_batch_tokens` clamped to 2048), so the 130.4 figure does
  not apply to 2 slots.

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
