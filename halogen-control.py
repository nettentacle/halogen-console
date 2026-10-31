#!/usr/bin/env python3
"""halogen / kyojin / strata / gsqhalo / hipfire / gufo 控制面板：局域网内远程启动 / 停止 / 重启模型服务，并显示实时系统状态。

设计要点
--------
* 只驱动 systemd 用户单元，不自己拼命令，这样启动方式只有一处定义：
  halogen 是 `halogen-docker.service`（2026-10-01 晚起跑在 Docker 里），
  kyojin 是 `kyojin-qwen.service`（2026-10-22 起仅 Qwen 车道；GLM 权重/单元已回收），
  strata 是 `strata.service`（2026-10-07 部署，Qwen3.8-Flash-Next GSQ-RCO IQ3_S；当晚由 UD-IQ4_XS 切换而来），
  gsqhalo 是 `gsqhalo.service`（2026-10-07 部署，podman 容器；Qwen3.8-Flash-Next GSQ-RCO IQ3_S
  + MTP/ngram 投机解码，来自 Aristo94/GSQHalo.cpp = strix-llama 的 GSQ 量化分支），
  hipfire 是 `hipfire-27b.service`（2026-10-17 部署，原生 Rust/HIP 引擎；Qwen3.8-27B MQ4 XTS
  + DFlash 草稿 + 视觉塔，端口 8731），
  gufo 是 `gufo.service`（2026-10-21 部署，gufo-org/gufo 官方预构建容器；Qwen3.8-Flash-Next
  UD-Q4_K_XL 4 分片 + MTP 草稿 + 视觉塔，端口 8731）。
* **六个引擎互斥**：各自都要独占本机大半内存（halogen/kyojin 约 105 GB GTT 常驻
  显存；strata 约 40 GB 显存 + 55 GiB 内存；gsqhalo 约 71 GB GTT；hipfire 27B 约 20–25 GB GTT；
  gufo 约 105 GiB GTT + 2 × 262144 上下文），
  同一时刻只能跑一个。
  面板因此把"启动/重启 X"实现为先停其余引擎再起 X（见 h_action），
  开机自启也做成互斥（启用一个自动取消其它，见 h_boot）。
* 无访问口令（局域网可信）。但状态变更接口要求 Content-Type 必须是
  application/json：这是非简单请求类型，跨站 fetch 想带它会先触发 CORS
  预检，而本服务不返回任何 CORS 头，所以别的网页无法诱导浏览器把模型
  停掉。浏览器直接打开本页面使用不受影响。
* systemctl 一律加 --no-block（切换时的"停另一个"除外，那一步要等停干净）：
  模型加载要几十秒到几分钟，接口不该阻塞，页面靠轮询 /api/status 看进度。
* 系统状态全部读 /proc 与 /sys，不依赖外部命令，也不引入新依赖。
  CPU 占用率要在两次采样之间求差，所以模块里留了一份上次的 /proc/stat。
  NPU（amdxdna）的利用率同理：驱动把"引擎忙碌时间"记在客户端的
  /proc/<pid>/fdinfo 里，两次采样求差即得（见 sysinfo 上方的 NPU 一节）。
  例外：2026-10-11 起 NPU 由 halogen 容器使用（root 进程），宿主上读别人的
  fdinfo 要同 uid 或 CAP_SYS_PTRACE，用户单元两者都没有 → 容器内 root 自己读、
  经一条常驻的 docker exec 流报给面板（docker 本就是面板读 halogen 日志的依赖）。
"""

import asyncio
import fcntl
import glob
import json
import logging
from contextlib import suppress
import os
import re
import shutil
import socket
import struct
import subprocess
import time
from pathlib import Path

from aiohttp import web, ClientSession, ClientTimeout

LISTEN_HOST = "0.0.0.0"
LISTEN_PORT = 8733
DOCKER = "/snap/bin/docker"  # 用户单元的 PATH 不含 /snap/bin，一律绝对路径
PODMAN = "/usr/bin/podman"   # gsqhalo 在 rootless podman 里（无守护进程）
GIB = 1024 ** 3

# 2026-10-29：halogen 开了 HALOGEN_API_KEY（除 /health 外全部路由要 `Authorization: Bearer <key>`）。
# 面板对所有引擎统一带上这个头——只有 halogen 认它，别的引擎忽略，省得每次按"这会儿谁在答话"判断。
# 密钥与单元挂进容器的是同一份宿主文件（600），读不到就返回空头（等于没开鉴权，照旧工作）。
_KEY_FILE = Path("/home/xiaoming/halogen-patch/halogen-api-key")


def _auth_headers():
    try:
        key = _KEY_FILE.read_text().strip()
    except OSError:
        return {}
    return {"Authorization": "Bearer " + key} if key else {}


def _get(session, url, headers=None):
    # 显式传 headers（GitHub 等外部请求）时不带本地引擎的鉴权头。
    return session.get(url, headers=_auth_headers() if headers is None else headers)


# ------------------------------------------------------------------ 引擎登记表
#
# 六个引擎互斥，同一时刻只应有一个在跑；所有引擎现在都绑定 8731（端口复用
# 正是互斥的结果，面板用 systemd 单元状态判断"当前是谁"，不再按端口区分）。
#   halogen: /health（JSON 状态）+ /metrics（prometheus 文本）
#   kyojin : /v1/models + /slots（JSON）+ /stats（serve.py 的本地补丁）
#   strata : /health（JSON 状态）+ /v1/status（JSON：版本 / 上下文 / 最近一次耗时）
# 2026-10-17：所有引擎统一到 8731。8080 仍是 llama.cpp 那一套的习惯端口，留给
# 手开的 llama-server/vLLM；原来的 8088 / 8081 / 11435 全部改为 8731。
ENGINES = {
    "halogen": {
        "unit": "halogen-docker",
        "label": "halogen",
        "model": "Qwen3.8-Flash-Next",
        "health": "http://127.0.0.1:8731/health",
        "metrics": "http://127.0.0.1:8731/metrics",
        "log_kind": "docker",
        "log_container": "halogen",
        # 检查点（2026-10-14 起）：halogen 是单模型引擎，破限/其它转换版不是新引擎、
        # 也不是另一个单元，而是同一个 halogen-docker.service 里 HALOGEN_CHECKPOINT
        # 指向的 .hgn 文件（都躺在 halogen-models 卷里，彼此同尺寸、同样互斥）。
        # 切换 = 原子改写单元文件那一行 + daemon-reload，随"启动/重启"生效（见 h_action）。
        "unit_file": "/home/xiaoming/.config/systemd/user/halogen-docker.service",
        "checkpoints": {
            "v2": {"file": "qwen38-flash-next-v2.hgn"},
            # Quat3rnion 的 OrcaRouter-Uncensored 原生 v2 转换；2026-10-14 实测
            # 拒答 4/10→0/10、其余门禁与 v2 同级（见 ~/halogen-patch/regress-0150/）
            # 2026-10-22 清盘删除，2026-10-30 重新下载并校验（sha256 b5bd925f…）
            "abliterated": {"file": "qwen38-flash-next-v2-abliterated.hgn"},
        },
    },
    "kyojin": {
        # 2026-10-22：GLM-5.3-Flash 车道与权重已回收，只保留 Qwen 车道单单元。
        "unit": "kyojin-qwen",
        "label": "kyojin",
        "model": "Qwen3.8-Flash-Next",
        "models": "http://127.0.0.1:8731/v1/models",
        "slots": "http://127.0.0.1:8731/slots",
        "stats": "http://127.0.0.1:8731/stats",
        # Qwen 车道有 /health（带 vision 标志）。
        "health": "http://127.0.0.1:8731/health",
        "log_kind": "journal",
        "log_unit": "kyojin-qwen",
        # 源码构建，没有版本号：用 git 提交当标识（面板显示短哈希 + 落后上游几个提交）
        "repo": "Yamz-Labs/kyojin",
        "dir": "/home/xiaoming/kyojin",
    },
    "strata": {
        "unit": "strata",
        "label": "strata",
        "model": "Qwen3.8-Flash-Next",
        "health": "http://127.0.0.1:8731/health",
        "stats": "http://127.0.0.1:8731/v1/status",
        "log_kind": "journal",
        "log_unit": "strata",
        # 有版本号（/v1/status 的 engine），但上游标签会落在检出之后（0.1.40 的检出
        # 已含 0.1.40.1 的修复），所以更新检查按 git 提交比，版本号只当显示名
        "repo": "Niko1221/Strata",
        "dir": "/home/xiaoming/Strata",
    },
    "gsqhalo": {
        "unit": "gsqhalo",
        "label": "gsqhalo",
        "model": "Qwen3.8-Flash-Next",
        # llama.cpp 系（GSQHalo.cpp），端口 8731：/health 判就绪、/slots 给上下文与忙闲、
        # /props 给模型路径、/metrics（--metrics 开启）给 prometheus 计数器。
        "health": "http://127.0.0.1:8731/health",
        "slots": "http://127.0.0.1:8731/slots",
        "props": "http://127.0.0.1:8731/props",
        "metrics": "http://127.0.0.1:8731/metrics",
        "log_kind": "podman",
        "log_container": "gsqhalo",
        # 源码构建，没有版本号：用 git 提交当标识（面板显示短哈希 + 落后上游几个提交）
        "repo": "Aristo94/GSQHalo.cpp",
        "dir": "/home/xiaoming/GSQHalo.cpp",
    },
    "hipfire": {
        "unit": "hipfire-27b",
        "label": "hipfire",
        "model": "Qwen3.8-27B MQ4 XTS",
        # 原生 Rust/HIP 引擎，端口 8731；/health 判就绪（loading_model 非空=加载中），
        # /stats 给状态轮询，/metrics 给累计计数与每请求 prefill/decode 直方图（面板指标源）。单元里 --vision + HIPFIRE_VISION_MODE=on
        # 打开 27B 视觉塔，所以面板的"图像输入"按已启用显示。
        "health": "http://127.0.0.1:8731/health",
        "stats": "http://127.0.0.1:8731/stats",
        "metrics": "http://127.0.0.1:8731/metrics",
        "log_kind": "journal",
        "log_unit": "hipfire-27b",
        "vision": True,
        # 单槽 262144 + DFlash + vision ≈ 22 GB GTT；给 40 GiB 余量
        "min_avail_gib": 40.0,
        # 有 tag 版本号（vX.Y.Z）；更新检查按 GitHub tags 比，不走 commit
        "repo": "warpfront/hipfire",
        "dir": "/home/xiaoming/.hipfire/src",
    },
    "gufo": {
        "unit": "gufo",
        "label": "gufo",
        "model": "Qwen3.8-Flash-Next",
        # gufo-org/gufo（Strix Halo 专用 C++/HIP 引擎，2026-10-21 部署）：官方预构建
        # 容器跑在 rootless podman 里，端口 8731。它的端点刻意与 llama.cpp 同名同形：
        # /health、/slots、/props、/metrics（llama.cpp prometheus 计数器 + gufo_* 扩展），
        # 所以状态/指标解析与 gsqhalo 同源；/props 的 model_alias 是它独有的标记。
        "health": "http://127.0.0.1:8731/health",
        "slots": "http://127.0.0.1:8731/slots",
        "props": "http://127.0.0.1:8731/props",
        "metrics": "http://127.0.0.1:8731/metrics",
        "log_kind": "podman",
        "log_container": "gufo",
        "vision": True,
        # UD-Q4_K_XL 权重 104 GiB + MTP 2.6 GiB + 视觉塔 0.9 GiB + 2 × 262144 会话状态；
        # 实测加载后 GPU 侧约 105 GiB、宿主余量 15.6 GiB，所以启动门槛给 112 GiB。
        "min_avail_gib": 112.0,
        # 有 tag 版本号（vX.Y.Z，release-please 发版，几乎每天一个）；按 GitHub tags 比
        "repo": "gufo-org/gufo",
    },
}
ENGINE_KEYS = tuple(ENGINES)
DEFAULT_ENGINE = "halogen"


def engine_of(request, body=None) -> str:
    """请求指定的引擎：?engine= 查询串优先，其次 POST body，默认 halogen。"""
    key = (request.query.get("engine")
           or (body or {}).get("engine") or DEFAULT_ENGINE)
    return key if key in ENGINES else DEFAULT_ENGINE


# ---------------------------------------------------------------- 模型变体
#
# 同一台引擎树跑不同模型 = 模型变体（kyojin 的 GLM-5.3-Flash / Qwen3.8-Flash-Next）：
# 各自一个 systemd 单元、都开在 8731、同样吃满内存 —— 所以变体之间也互斥。
# 请求用 ?variant= 指定；没给 / 非法时回退到引擎注册表里的缺省单元（"unit" 字段）。
# 没有变体的引擎（halogen/strata）这些函数全部退化成默认行为。
def variants_of(engine: str) -> dict:
    return ENGINES.get(engine, {}).get("variants") or {}


def variant_key(engine: str, request=None, body=None):
    """请求点名的变体键；没给或非法时 None（=引擎默认变体）。"""
    known = variants_of(engine)
    if not known:
        return None
    key = ((request.query.get("variant") if request is not None else None)
           or (body or {}).get("variant"))
    return key if key in known else None


def unit_of(engine: str, variant=None) -> str:
    cfg = ENGINES[engine]
    if variant and variant in (cfg.get("variants") or {}):
        return cfg["variants"][variant]["unit"]
    return cfg["unit"]


async def active_variant(engine: str):
    """正在跑（或正在切换）的变体键；没有变体或都没跑时 None。"""
    for key, cfg in variants_of(engine).items():
        ao, *_ = await unit_probe(cfg["unit"])
        if ao in ("active", "activating", "deactivating"):
            return key
    return None


# ---------------------------------------------------------------- 检查点（halogen）
#
# halogen 的"破限版"不是另一个引擎、也不是另一个单元：同一个 halogen-docker.service，
# 换 ExecStart 里 -e HALOGEN_CHECKPOINT=/models/<file> 指向的 .hgn 文件（同尺寸、
# 同内存档位，天然互斥）。注册表在服务端（前端只发键名），切换 = 原子改写单元文件
# 那一行 + daemon-reload，由随后的"启动/重启"生效——注意：改文件对已加载的引擎无效，
# 所以点名了"与在跑不同"的检查点时，即使点"启动"也要先停当前实例（h_action 按此处理）。
_CP_LINE_RE = re.compile(
    r"^(\s*-e HALOGEN_CHECKPOINT=)(/models/)([^\s\\]+)(\s*\\)\s*$", re.M)

# 滚动备份：每次改文件前留一份上一版（固定名，只保最近一次；单元文件自己的
# 注释区还留着历次变更记录，长期历史不靠这个文件）
BACKUP_DIR = Path("/home/xiaoming/halogen-patch")


def checkpoints_of(engine: str) -> dict:
    return ENGINES.get(engine, {}).get("checkpoints") or {}


def checkpoint_key(engine: str, request=None, body=None):
    """请求点名的检查点键；没给、非法或引擎无注册表时 None（=不动）。"""
    known = checkpoints_of(engine)
    if not known:
        return None
    key = ((request.query.get("checkpoint") if request is not None else None)
           or (body or {}).get("checkpoint"))
    return key if key in known else None


def checkpoint_current_file(engine: str):
    """单元文件里 HALOGEN_CHECKPOINT 现在的文件名；读不到或没这行返回 None。"""
    path = ENGINES.get(engine, {}).get("unit_file")
    if not path:
        return None
    try:
        m = _CP_LINE_RE.search(Path(path).read_text(encoding="utf-8"))
    except OSError:
        return None
    return m.group(3) if m else None


def checkpoint_key_now(engine: str):
    """当前单元文件对应的检查点键；未知文件名原样返回（前端按未知显示）。"""
    name = checkpoint_current_file(engine)
    if not name:
        return None
    for k, c in checkpoints_of(engine).items():
        if c.get("file") == name:
            return k
    return name


def checkpoint_write(engine: str, key: str):
    """把单元文件的检查点行改写成 key 对应文件（原子写 + 滚动备份 + 注释留痕）。

    返回 (changed, error)：文件本来就指这个 key 时 changed=False、error=None。
    """
    fname = (checkpoints_of(engine).get(key) or {}).get("file")
    path = Path(ENGINES[engine]["unit_file"])
    try:
        src = path.read_text(encoding="utf-8")
    except OSError as e:
        return False, f"读单元文件失败：{e}"
    m = _CP_LINE_RE.search(src)
    if not m:
        return False, "单元文件里找不到 HALOGEN_CHECKPOINT 行"
    if m.group(3) == fname:
        return False, None
    with suppress(OSError):
        shutil.copy2(path, BACKUP_DIR / f"{path.name}.pre-checkpoint")
    new = src.replace(
        m.group(0),
        f"{m.group(1)}{m.group(2)}{fname}{m.group(4)}")
    # 变更留痕：插进文件头注释区的 ExecStart 之前（本文件一贯的变更记录风格）
    i = new.find("\nExecStart=")
    if i >= 0:
        note = (f"# {time.strftime('%Y-%m-%d %H:%M')} 面板(8733)：检查点 "
                f"{m.group(3)} → {fname}\n")
        new = new[:i + 1] + note + new[i + 1:]
    tmp = path.with_name(path.name + ".panel-tmp")
    try:
        tmp.write_text(new, encoding="utf-8")
        os.replace(tmp, path)
    except OSError as e:
        with suppress(OSError):
            tmp.unlink()
        return False, f"写单元文件失败：{e}"
    return True, None


# 正在运行实例的检查点：容器的 env 是运行事实，单元文件只是"下次启动会用什么"。
# 二者不一致（切换过但没重启）正是前端"待切换"提示要展示的状态。
# docker inspect 每个实例只问一遍：键是单元启动时刻（与 cache_headroom 同模式）。
_cp_run_cache = {"key": None, "data": None}


async def checkpoint_running(engine, start_key, engine_ready):
    if engine != "halogen" or not start_key or not engine_ready:
        return None
    if _cp_run_cache["key"] == start_key:
        return _cp_run_cache["data"]
    rc, out = await run(DOCKER, "inspect", ENGINES["halogen"]["log_container"],
                        "--format", "{{json .Config.Env}}", timeout=15)
    if rc != 0:
        return None
    try:
        env = json.loads(out)
        ref = next((e.split("=", 1)[1] for e in env
                    if e.startswith("HALOGEN_CHECKPOINT=")), None)
        if not ref:
            return None
        name = ref.rsplit("/", 1)[-1]
        key = next((k for k, c in checkpoints_of(engine).items()
                    if c.get("file") == name), name)
    except (ValueError, AttributeError, TypeError):
        return None
    _cp_run_cache["key"] = start_key
    _cp_run_cache["data"] = key
    return key

# 面板自身探测产生的访问日志行，默认从日志视图里滤掉
# （stats 是 2026-10-08 起加的探测端点：halogen 没有它，切换过渡期或状态不明时
# 的探测会在这里留下 404，与健康/指标探测同属一类噪声）
_PROBE_NOISE = re.compile(
    # docker/llama.cpp 访问日志形态：`"GET /health HTTP/1.1" 200`
    r'\s"(?:GET|HEAD) /(?:health|metrics|stats|cache)\b[^"]*"\s+\d{3}'
    # gufo 形态：`method=GET path=/metrics ...`（1 Hz 探测，不滤会刷屏）
    r'|\bmethod=(?:GET|HEAD) path=/(?:health|metrics|stats|cache|slots|props)\b')

# ---------------------------------------------------------------- 界面语言
#
# 页面上的中英文切换在前端做，但会被用户看到的那几条报错在服务端拼，
# 所以前端每次请求都带 X-Lang 头（见 html 里的 post()）。只翻译会显示
# 在页面上的文案；服务端自己的日志仍然是中文。
_MSG = {
    "ctype": {"zh": "Content-Type 必须是 application/json",
              "en": "Content-Type must be application/json"},
    "action": {"zh": "action 必须是 start/stop/restart",
               "en": "action must be start/stop/restart"},
    "engine": {"zh": "engine 必须是 halogen、kyojin、strata、gsqhalo、hipfire 或 gufo",
               "en": "engine must be halogen, kyojin, strata, gsqhalo, hipfire or gufo"},
    "rc": {"zh": "systemctl 退出码 {rc}", "en": "systemctl exited with code {rc}"},
}


def lang_of(request) -> str:
    """X-Lang 优先（前端语言开关），其次 Accept-Language，默认中文。"""
    tag = (request.headers.get("X-Lang")
           or request.headers.get("Accept-Language") or "zh").strip().lower()
    return "en" if tag.startswith("en") else "zh"


def msg(request, key, **kw) -> str:
    text = _MSG[key][lang_of(request)]
    return text.format(**kw) if kw else text


# ---------------------------------------------------------------- 命令执行


async def run(*argv, timeout=15):
    """跑一条命令，返回 (returncode, stdout+stderr 去尾空白)。"""
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return proc.returncode, out.decode(errors="replace").strip()
    except asyncio.TimeoutError:
        return 124, f"timeout after {timeout}s"
    except Exception as e:  # noqa: BLE001
        return 1, f"{type(e).__name__}: {e}"


async def unit_probe(unit: str):
    """给一个具体单元名问状态（unit_state 的底层；kyojin 的模型变体各有单元）。"""
    _, active = await run("systemctl", "--user", "is-active", unit)
    _, enabled = await run("systemctl", "--user", "is-enabled", unit)
    _, mono = await run("systemctl", "--user", "show", unit,
                        "-p", "ActiveEnterTimestampMonotonic", "--value")
    uptime = None
    try:
        boot = float(Path("/proc/uptime").read_text().split()[0])
        m = int(mono)
        if m > 0 and active == "active":
            uptime = max(0, int(boot - m / 1_000_000))
    except Exception:  # noqa: BLE001
        pass
    # 第三个返回值是本次运行的启动时刻，用作"每次启动才变的数据"的缓存键
    return active or "unknown", enabled or "unknown", uptime, (mono or "").strip()


async def unit_state(engine: str = DEFAULT_ENGINE):
    return await unit_probe(ENGINES[engine]["unit"])


# ------------------------------------------------- 启动时才有的关键数字
#
# 引擎在启动时打印 `host memory left for everything else: N contiguous 2 MiB
# blocks (X GiB total)` —— 这是本机最关键的健康指标：低于约 10 GiB 时查找表
# 会从磁盘反复换入，prefill 从秒变分钟。它只在启动日志里出现一次，/health
# 也不暴露，所以这里解析出来并缓存：同一引擎实例内不会变，只在重启时重取。
_headroom_cache = {"key": None, "data": None}

_HEADROOM_RE = re.compile(
    r"host memory left for everything else:\s*(\d+)\s+contiguous 2 MiB blocks"
    r"\s*\(([\d.]+) GiB total")

# hipfire 没有 halogen 那条 contiguous 2 MiB 行；它在每次 load 时打印
# `max_seq budget: free_after_weights=<bytes> ...`，作为"权重后可用"的等价指标。
_HIPFIRE_HEADROOM_RE = re.compile(r"max_seq budget: free_after_weights=(\d+)")

# gufo 在 load_completed 行里报加载后的宿主可用内存（host_available_mib），
# 即"权重与缓存预留之后还剩给系统/其它进程多少"——与 halogen 的页缓存余量同义。
_GUFO_HEADROOM_RE = re.compile(r"load_completed\b.*host_available_mib=(\d+)")


async def cache_headroom(engine, start_key, engine_ready):
    # halogen：页缓存余量（contiguous 2 MiB blocks）；hipfire：启动日志的
    # free_after_weights（权重后可用）；gufo：load_completed 的 host_available_mib。
    # 三者口径不同，前端按 source 换标签。
    if engine == "gufo":
        if not start_key:
            return None
        if _headroom_cache["key"] == start_key:
            return _headroom_cache["data"]
        if not engine_ready:
            return None
        _, out = await run(PODMAN, "logs", ENGINES["gufo"]["log_container"], timeout=20)
        data = None
        for line in out.splitlines():
            m = _GUFO_HEADROOM_RE.search(line)
            if m:
                data = {"gib": round(int(m.group(1)) / 1024, 1), "blocks": None,
                        "source": "host_available"}
        if data is not None:
            _headroom_cache["key"] = start_key
            _headroom_cache["data"] = data
        return data
    if engine == "hipfire":
        if not start_key:
            return None
        if _headroom_cache["key"] == start_key:
            return _headroom_cache["data"]
        if not engine_ready:
            return None
        _, out = await run("journalctl", "--user", "-u", "hipfire-27b",
                           "-n", "500", "--no-pager", "-o", "cat", timeout=20)
        data = None
        for line in out.splitlines():
            m = _HIPFIRE_HEADROOM_RE.search(line)
            if m:
                data = {"gib": round(int(m.group(1)) / 1073741824, 1),
                        "blocks": None, "source": "free_after_weights"}
        if data is not None:
            _headroom_cache["key"] = start_key
            _headroom_cache["data"] = data
        return data
    if engine != "halogen" or not start_key:
        return None
    if _headroom_cache["key"] == start_key:
        return _headroom_cache["data"]
    # 这行日志要等引擎加载到约 3 秒才写出，而单元一 active 就可能被问到。
    # 就绪意味着 socket 已打开，必然在那行之后，此时扫到的才是本轮的；
    # 否则扫到的是上一轮的行，缓存住会让本轮一直显示旧值。
    if not engine_ready:
        return None
    # Docker 版：引擎日志在容器的 json-file 里（--rm 每次重建容器，所以
    # `docker logs` 天然只覆盖当前这一轮实例，正合 start_key 的语义）。
    _, out = await run(DOCKER, "logs", ENGINES["halogen"]["log_container"], timeout=20)
    data = None
    seen = set()
    for line in out.splitlines():
        if line in seen:
            continue
        seen.add(line)
        m = _HEADROOM_RE.search(line)
        if m:
            data = {"blocks": int(m.group(1)), "gib": float(m.group(2))}
    # 没扫到就不缓存：留一次重试的机会，而不是把空值钉死在这一轮的键上。
    if data is not None:
        _headroom_cache["key"] = start_key
        _headroom_cache["data"] = data
    return data


# halogen 每次运行（start_key=ActiveEnterTimestampMonotonic）是否曾报告过就绪。
# 用途：一次运行里「曾经就绪、现在问不到」= 引擎无响应（典型是内核内存压缩），
# 不是模型在加载；只有从未就绪过（首启）才显示"正在加载模型…"。
_ready_seen = {}


async def engine_state(engine: str = DEFAULT_ENGINE, start_key: str = None):
    """引擎自己的就绪状态：服务 active ≠ 模型已加载完。"""
    if engine == "hipfire":
        return await engine_state_hipfire()
    if engine == "kyojin":
        return await engine_state_kyojin()
    if engine == "strata":
        return await engine_state_strata()
    if engine == "gsqhalo":
        return await engine_state_gsqhalo()
    if engine == "gufo":
        return await engine_state_gufo()
    try:
        timeout = ClientTimeout(total=4)
        async with ClientSession(timeout=timeout) as s:
            async with _get(s, ENGINES["halogen"]["health"]) as r:
                d = await r.json()
                # 8731 现在所有引擎共用：只认 halogen 专属的 version 字典，
                # 避免把别家 /health 误当成 halogen。
                ver = d.get("version")
                if not isinstance(ver, dict) or not ver.get("engine"):
                    return {"ready": False, "status": "unreachable"}
                ready = d.get("status") == "ok"
                if ready and start_key:
                    _ready_seen[engine] = start_key
                return {
                    "ready": ready,
                    "status": d.get("status"),
                    "version": ver.get("engine"),
                    "model": d.get("model"),
                    "vision": d.get("vision", {}).get("enabled"),
                    "kv_pool": d.get("kv_pool_positions"),
                    "slots": d.get("slots"),
                    "busy": d.get("busy"),
                }
    except Exception:  # noqa: BLE001
        # 本次运行里曾就绪过 → 现在连不上/超时 = 无响应（内存压缩会卡住 PING），
        # 不是重新加载模型。只有从未就绪过才报 unreachable（前端显示加载中）。
        if start_key and _ready_seen.get(engine) == start_key:
            return {"ready": False, "status": "engine_unresponsive"}
        return {"ready": False, "status": "unreachable"}


async def engine_state_kyojin():
    """kyojin 没有统一 /health：端口打开即模型加载+调优完成，用 /v1/models 判就绪，
    /slots 补忙闲与上下文窗口。就绪之前 8731 拒连，这里表现为 unreachable。

    视觉能力（2026-10-07 补）：Qwen 车道有 /health 且带 vision 标志（实测 true），读它；
    GLM 车道没有 /health、代码里也没有任何图像输入路径 → 按"关"显示（不留 — 那种"未知"）。

    head：本地 git 短提交号——引擎没跑时也能给出，面板拿它当"版本"显示。"""
    head = await git_head("kyojin")
    try:
        timeout = ClientTimeout(total=4)
        async with ClientSession(timeout=timeout) as s:
            async with _get(s, ENGINES["kyojin"]["models"]) as r:
                d = await r.json()
        model = None
        data = d.get("data") or []
        if data:
            model = data[0].get("id")
        vision = False                            # GLM 车道：无 /health（404）→ 关
        try:
            async with ClientSession(timeout=ClientTimeout(total=4)) as s:
                async with _get(s, ENGINES["kyojin"]["health"]) as hr:
                    if hr.status == 200:
                        vision = bool((await hr.json()).get("vision"))
        except Exception:  # noqa: BLE001
            pass
        ctx = busy = None
        slots = None
        try:
            async with ClientSession(timeout=ClientTimeout(total=4)) as s:
                async with _get(s, ENGINES["kyojin"]["slots"]) as r:
                    slots = await r.json()
            if isinstance(slots, list) and slots:
                ctx = slots[0].get("n_ctx")
                busy = any(bool(sl.get("is_processing")) for sl in slots)
        except Exception:  # noqa: BLE001
            pass
        # 8731 共用：/slots 是 kyojin 的必备端点；halogen/strata/hipfire 没有它，
        # 不能把别家 /v1/models 当 kyojin。
        if not isinstance(slots, list):
            return {"ready": False, "status": "unreachable", "head": head}
        return {"ready": True, "status": "ok", "model": model, "version": None,
                "vision": vision, "kv_pool": ctx, "slots": None, "busy": busy, "head": head}
    except Exception:  # noqa: BLE001
        return {"ready": False, "status": "unreachable", "head": head}


async def engine_state_strata():
    """strata 现在也在 8731（全引擎统一）：/health 判就绪与视觉（loaded / images），
    /v1/status 补版本号、上下文窗口与并发槽数。共用端口下必须要求 /v1/status 是
    strata 的 JSON，否则别家 /health 会被误读。"""
    try:
        timeout = ClientTimeout(total=4)
        async with ClientSession(timeout=timeout) as s:
            async with _get(s, ENGINES["strata"]["health"]) as r:
                h = await r.json()
            async with _get(s, ENGINES["strata"]["stats"]) as r:
                st = await r.json()
        if not isinstance(st, dict) or not st.get("engine"):
            return {"ready": False, "status": "unreachable"}
        return {
            "ready": bool(h.get("loaded")),
            "status": h.get("status"),
            "version": st.get("engine"),
            "model": h.get("model") or st.get("model"),
            "vision": h.get("images"),
            "kv_pool": (st.get("context") or {}).get("max_positions"),
            "slots": (st.get("concurrency") or {}).get("serving"),
            "busy": bool((st.get("activity") or {}).get("in_flight")),
        }
    except Exception:  # noqa: BLE001
        return {"ready": False, "status": "unreachable"}


async def engine_state_gsqhalo():
    """gsqhalo（llama.cpp 系，现在也在 8731）：/health 判就绪，/slots 补上下文窗口与忙闲
    （kv_unified 下每个槽都报同一个池大小，取第一个；槽数单独给），/props 取模型
    文件名与视觉能力（加载了 -mm 视觉塔时 modalities.vision=true；没加载就是关）。
    共用端口下必须要求 /props 是 gsqhalo 的响应，否则 halogen 的 /health 会被误读。

    head：本地 git 短提交号——引擎没跑时也能给出，面板拿它当"版本"显示。"""
    head = await git_head("gsqhalo")
    try:
        timeout = ClientTimeout(total=4)
        async with ClientSession(timeout=timeout) as s:
            async with _get(s, ENGINES["gsqhalo"]["health"]) as r:
                h = await r.json()
        model = None
        vision = False
        props_ok = False
        try:
            async with ClientSession(timeout=ClientTimeout(total=4)) as s:
                async with _get(s, ENGINES["gsqhalo"]["props"]) as r:
                    p = await r.json()
            props_ok = isinstance(p, dict)
            mp = (p or {}).get("model_path") or ""
            if mp:
                base = mp.replace("\\", "/").rsplit("/", 1)[-1]
                # 分片文件名 → 可读名：去掉 "-00001-of-00002" 分片号与 .gguf 后缀
                model = re.sub(r"-\d{5}-of-\d{5}(?=\.gguf$)", "", base)
                if model.endswith(".gguf"):
                    model = model[:-len(".gguf")]
            vision = bool(((p or {}).get("modalities") or {}).get("vision"))
        except Exception:  # noqa: BLE001
            pass
        if not props_ok:
            return {"ready": False, "status": "unreachable", "head": head}
        ctx = busy = nslots = None
        try:
            async with ClientSession(timeout=ClientTimeout(total=4)) as s:
                async with _get(s, ENGINES["gsqhalo"]["slots"]) as r:
                    slots = await r.json()
            if isinstance(slots, list) and slots:
                ctx = slots[0].get("n_ctx")
                nslots = len(slots)
                busy = any(bool(sl.get("is_processing")) for sl in slots)
        except Exception:  # noqa: BLE001
            pass
        return {"ready": h.get("status") == "ok", "status": h.get("status"),
                "version": None, "model": model, "vision": vision,
                # kv_pool/slots 的口径对齐 halogen：池位置数 / 并发槽数
                "kv_pool": ctx, "slots": nslots, "busy": busy, "head": head}
    except Exception:  # noqa: BLE001
        return {"ready": False, "status": "unreachable", "head": head}


async def engine_state_gufo():
    """gufo（官方预构建容器，8731）：端点刻意与 llama.cpp 同形，但 /props 的
    model_alias 是它独有的身份标记（共用端口下用它排除别家的响应）。
    /props 给 build_info（0.20.0 (0c03748)）、槽数与每槽上下文、视觉能力；
    /slots 补忙闲；/health 判就绪。版本取 build_info 里的版本号，供 tag 更新检查用。"""
    try:
        timeout = ClientTimeout(total=4)
        async with ClientSession(timeout=timeout) as s:
            async with _get(s, ENGINES["gufo"]["props"]) as r:
                p = await r.json()
        if not isinstance(p, dict) or "model_alias" not in p:
            return {"ready": False, "status": "unreachable"}
        ready = False
        try:
            async with ClientSession(timeout=ClientTimeout(total=4)) as s:
                async with _get(s, ENGINES["gufo"]["health"]) as r:
                    ready = (await r.json()).get("status") == "ok"
        except Exception:  # noqa: BLE001
            pass
        nslots = p.get("total_slots")
        ctx = busy = None
        try:
            async with ClientSession(timeout=ClientTimeout(total=4)) as s:
                async with _get(s, ENGINES["gufo"]["slots"]) as r:
                    slots = await r.json()
            if isinstance(slots, list) and slots:
                ctx = slots[0].get("n_ctx")
                nslots = len(slots)
                busy = any(bool(sl.get("is_processing")) for sl in slots)
        except Exception:  # noqa: BLE001
            pass
        build = p.get("build_info") or ""
        m = re.match(r"(\d+\.\d+\.\d+)", build)
        return {
            "ready": ready,
            "status": "ok" if ready else "loading",
            "version": m.group(1) if m else None,
            "model": p.get("model_alias") or p.get("model"),
            "vision": bool((p.get("modalities") or {}).get("vision")),
            # 与 gsqhalo 同口径：kv_pool = 每槽上下文，slots = 槽数
            "kv_pool": ctx,
            "slots": nslots,
            "busy": busy,
        }
    except Exception:  # noqa: BLE001
        return {"ready": False, "status": "unreachable"}


async def engine_state_hipfire():
    """hipfire（原生 Rust/HIP，现在也在 8731）：/health 判就绪（loading_model 非空=加载中），
    /stats 补队列深度与请求数。版本取本地 tag（v0.4.2.1），tag 不在则退回 git 短哈希。
    视觉能力来自单元配置（--vision + HIPFIRE_VISION_MODE=on），/health 不暴露该字段；
    槽数从 capabilities.multi_slot_slots 取（当前单元是单槽 262144；开 multi 时自动读 2+）。"""
    head = await git_head("hipfire")
    version = (await git_tag("hipfire") or "").lstrip("vV") or None
    try:
        timeout = ClientTimeout(total=4)
        async with ClientSession(timeout=timeout) as s:
            async with _get(s, ENGINES["hipfire"]["health"]) as r:
                h = await r.json()
            stats = None
            try:
                async with _get(s, ENGINES["hipfire"]["stats"]) as r:
                    stats = await r.json()
            except Exception:  # noqa: BLE001
                pass
        # 8731 共用：/stats 是 hipfire 的必备端点（requests_served），halogen 的
        # /health 没有它，不能把别家响应当 hipfire。
        if not isinstance(stats, dict) or "requests_served" not in stats:
            return {"ready": False, "status": "unreachable", "version": version,
                    "head": head}
        loading = h.get("loading_model")
        ready = h.get("status") == "ok" and bool(h.get("model")) and not loading
        caps = h.get("capabilities") or {}
        slots = caps.get("multi_slot_slots") if caps.get("multi_slot") else 1
        return {
            "ready": ready,
            "status": "loading" if loading else h.get("status"),
            "version": version,
            "model": h.get("model") or loading,
            "vision": bool(ENGINES["hipfire"].get("vision")),
            "kv_pool": h.get("n_ctx"),
            "slots": slots or 1,
            "busy": bool((stats or {}).get("queue_depth")),
            "head": head,
        }
    except Exception:  # noqa: BLE001
        return {"ready": False, "status": "unreachable", "version": version,
                "head": head}


# ------------------------------------------------------------ 推理速度采集

_metrics_prev = None
_metrics_src = None          # _metrics_prev 的来源引擎（切引擎 = 计数器换源）
_last_rates = None
# 每次真正测到一次请求速度就往这里追加一条，供 KPI 卡片画吞吐趋势
_rate_history = []
RATE_HIST_LEN = 40
# 本次模型加载以来的单次请求速率峰值（引擎重启时归零）
_rate_peak = {"prefill": None, "decode": None}
# 最近一次"真正测到"的速率。缓存全命中的请求不会产生预填充测量值，
# 若直接用最新一次采样，卡片会突然变成空值，所以这里保留最后一个有效值。
_last_good = {"prefill": None, "decode": None}


def _reset_rates():
    global _last_rates
    _last_rates = None
    _rate_history.clear()
    _rate_peak.update(prefill=None, decode=None)
    _last_good.update(prefill=None, decode=None)


def _settle_rates(src, now, counters):
    """引擎间共用的速率结算：对累计计数器两次采样求差。

    计数器在请求完成后结算；差值是采样窗口内完成请求的加权平均速度，
    不是请求内部的瞬时速度。使用累计计数器，避免其它 scraper 清空区间
    gauge 的影响。引擎切换（换源）或计数器倒退按"重启"处理，清掉采样。
    """
    global _metrics_prev, _metrics_src
    cur = (now, *counters)
    live_p = live_d = None
    consume = True          # 是否把本次采样登记为新的基准
    prev = _metrics_prev
    if prev is not None and _metrics_src != src:
        _reset_rates()      # 换了引擎，两边的计数器不可比
        prev = None
    have_both = prev is not None and all(x is not None for x in prev[1:]) and all(x is not None for x in cur[1:])
    if have_both:
        dt = cur[0] - prev[0]
        # 计数器倒退 = 引擎重启过，之前的采样作废
        if cur[1] < prev[1] or cur[3] < prev[3]:
            _reset_rates()
        elif dt < 0.5:
            # 窗口太短，不消费这次采样，把差值累积给下一次。
            # 多个客户端同时轮询时会出现这种短窗口；若直接丢掉，
            # 请求完成那一瞬的计数跳变就漏了。
            consume = False
        elif dt <= 120:
            dpt, dps = cur[1] - prev[1], cur[2] - prev[2]
            dtt, dts = cur[3] - prev[3], cur[4] - prev[4]
            if dpt > 0 and dps > 0.05:
                live_p = round(dpt / dps, 1)
            if dtt > 0 and dts > 0.05:
                live_d = round(dtt / dts, 1)
            if live_p is not None or live_d is not None:
                _last_rates = {"prefill": live_p, "decode": live_d, "at": time.time()}
                _rate_history.append({
                    "t": int(time.time()), "p": live_p, "d": live_d,
                    "pt": int(dpt), "ct": int(dtt),   # 该次请求处理的提示/生成 token 数
                })
                if len(_rate_history) > RATE_HIST_LEN:
                    del _rate_history[:-RATE_HIST_LEN]
                for k, v in (("prefill", live_p), ("decode", live_d)):
                    if v is not None:
                        if _rate_peak[k] is None or v > _rate_peak[k]:
                            _rate_peak[k] = v
                        _last_good[k] = v
    if consume:
        _metrics_prev = cur
        _metrics_src = src
    return live_p, live_d


def _now_fields(live_p, live_d, measured_at):
    return {
        "prefill_now": live_p if live_p is not None else _last_good["prefill"],
        "decode_now": live_d if live_d is not None else _last_good["decode"],
        "is_live": False,
        "measured_at": measured_at,
        "rate_mode": "completed_requests",
        # 峰值是"单次请求速率"的最大值：引擎计数器按请求原子结算，
        # 拿不到请求内部的瞬时值，这是能做到的最接近的口径。
        "prefill_peak": _rate_peak["prefill"],
        "decode_peak": _rate_peak["decode"],
    }


def _div(a, b):
    return round(a / b, 1) if a is not None and b else None


def _metrics_halogen(txt):
    """halogen 的 /metrics（prometheus 文本）→ 面板指标。"""
    v = {}
    for line in txt.splitlines():
        if not line or line.startswith("#"):
            continue
        name, _, val = line.rpartition(" ")
        try:
            v[name] = float(val)
        except ValueError:
            continue

    g = v.get
    live_p, live_d = _settle_rates(
        "halogen", time.monotonic(),
        (g("llamacpp:prompt_tokens_total"), g("llamacpp:prompt_seconds_total"),
         g("llamacpp:tokens_predicted_total"), g("llamacpp:tokens_predicted_seconds_total")))

    last = _last_rates or {}
    dr, da = g("halogen:draft_tokens_total"), g("halogen:draft_tokens_accepted_total")
    cached = g("halogen:prompt_tokens_cached_total")
    proc = g("llamacpp:prompt_tokens_total")

    out = {
        "engine": "halogen",
        "prefill_avg": _div(g("llamacpp:prompt_tokens_total"), g("llamacpp:prompt_seconds_total")),
        "decode_avg": _div(g("llamacpp:tokens_predicted_total"), g("llamacpp:tokens_predicted_seconds_total")),
        "processing": g("llamacpp:requests_processing"),
        "deferred": g("llamacpp:requests_deferred"),
        "draft_accept_pct": round(da * 100.0 / dr, 1) if dr and da is not None else None,
        "cache_hit_pct": round(cached * 100.0 / (cached + proc), 1) if cached is not None and proc and (cached + proc) > 0 else None,
        "kv_used": g("llamacpp:kv_cache_tokens"),
        "kv_ratio_pct": round(g("llamacpp:kv_cache_usage_ratio") * 100, 1) if g("llamacpp:kv_cache_usage_ratio") is not None else None,
        "requests_total": g("halogen:requests_total"),
    }
    out.update(_now_fields(live_p, live_d, last.get("at")))
    return out


def _metrics_kyojin(txt):
    """kyojin 的 /stats（serve.py 本地补丁的 JSON）→ 面板指标。

    口径与 halogen 对齐：速度为"完成请求的加权平均"，平均/总数来自累计
    计数器。MTP 草稿采纳率来自 serve.py 本地补丁的 draft_tokens/draft_accepted
    （引擎口径 accepted/(accepted+rejected)；没打该补丁的引擎此键缺席、字段为 null）；
    KV 占用该端点仍没有，前端显示 —。
    """
    d = json.loads(txt)
    # serving 是扣除预热基线后的"真实流量"，优先用它；老补丁只有 cumulative。
    cum = d.get("serving") or d.get("cumulative") or {}
    last = d.get("last") or {}
    pt, ps = cum.get("prompt_tokens"), cum.get("prompt_seconds")
    gt, gs = cum.get("predicted_tokens"), cum.get("predicted_seconds")
    live_p, live_d = _settle_rates("kyojin", time.monotonic(), (pt, ps, gt, gs))

    cached = cum.get("cached_tokens") or 0
    total_prompt = cum.get("prompt_tokens") or 0
    draft, draft_acc = cum.get("draft_tokens") or 0, cum.get("draft_accepted") or 0
    out = {
        "engine": "kyojin",
        "prefill_avg": _div(pt, ps),
        "decode_avg": _div(gt, gs),
        "processing": None,          # /stats 不暴露在途请求数；忙闲由 /slots 提供
        "deferred": None,
        "draft_accept_pct": round(draft_acc * 100.0 / draft, 1) if draft else None,
        "cache_hit_pct": round(cached * 100.0 / (cached + total_prompt), 1) if (cached + total_prompt) > 0 else None,
        "kv_used": None,
        "kv_ratio_pct": None,
        "requests_total": cum.get("requests"),
    }
    out.update(_now_fields(live_p, live_d, last.get("at") or (_last_rates or {}).get("at")))
    return out


# strata 的 /v1/status 只给"最近一次请求"的耗时（last_timings，llama.cpp 命名）与
# 累计请求数，没有累计 token/秒计数器。面板按每次完成的请求自己累积（引擎
# `started` 变了就是重启，清零），口径与另两个引擎一致：平均/峰值/趋势都来自
# "完成的请求"。last_timings 在两次请求之间不变，所以它天然是"最近完成"值。
_strata_acc = {"key": None, "last_at": None, "pt": 0.0, "ps": 0.0, "gt": 0.0,
               "gs": 0.0, "draft": 0.0, "draft_acc": 0.0, "cached": 0.0,
               "processed": 0.0, "peak_p": None, "peak_d": None}


def _metrics_strata(txt):
    """strata 的 /v1/status（JSON）→ 面板指标。

    prompt_n 是本次实际读的提示（不含缓存复用），cache_n 是会话缓存已经持有的
    部分、prompt_ms 只覆盖 prompt_n —— 与 halogen 的口径一致（平均速度用实读
    部分）；趋势表的"提示 tokens"给整段提示（prompt_n + cache_n）。
    """
    d = json.loads(txt)
    # 8731（halogen/kyojin）↔ 8731（strata）换源时清掉共享的采样基准与趋势
    _settle_rates("strata", time.monotonic(), (None, None, None, None))
    t = d.get("last_timings") or {}
    act = d.get("activity") or {}
    acc = _strata_acc
    if acc["key"] != d.get("started"):          # 引擎重启：本面板的累计清零
        acc.update(key=d.get("started"), last_at=None, pt=0.0, ps=0.0, gt=0.0,
                   gs=0.0, draft=0.0, draft_acc=0.0, cached=0.0, processed=0.0,
                   peak_p=None, peak_d=None)
        _rate_history.clear()                   # 趋势/采样表也是这一轮实例的，别留上一轮的

    def num(v):
        return float(v) if isinstance(v, (int, float)) else 0.0

    at = t.get("at")
    if at is not None and at != acc["last_at"]:  # 有新完成的请求（秒级时间戳换新）
        acc["last_at"] = at
        pn, pms = num(t.get("prompt_n")), num(t.get("prompt_ms"))
        gn, gms = num(t.get("predicted_n")), num(t.get("predicted_ms"))
        cn = num(t.get("cache_n"))
        acc["pt"] += pn
        acc["ps"] += pms / 1000.0
        acc["gt"] += gn
        acc["gs"] += gms / 1000.0
        acc["cached"] += cn
        acc["processed"] += pn
        acc["draft"] += num(t.get("draft_n"))
        acc["draft_acc"] += num(t.get("draft_n_accepted"))
        p, dec = t.get("prompt_per_second"), t.get("predicted_per_second")
        if p:
            acc["peak_p"] = p if acc["peak_p"] is None else max(acc["peak_p"], p)
        if dec:
            acc["peak_d"] = dec if acc["peak_d"] is None else max(acc["peak_d"], dec)
        _rate_history.append({"t": int(time.time()), "p": p, "d": dec,
                              "pt": int(pn + cn), "ct": int(gn)})
        if len(_rate_history) > RATE_HIST_LEN:
            del _rate_history[:-RATE_HIST_LEN]

    return {
        "engine": "strata",
        "prefill_avg": _div(acc["pt"], acc["ps"]) if acc["pt"] else None,
        "decode_avg": _div(acc["gt"], acc["gs"]) if acc["gt"] else None,
        "processing": act.get("in_flight"),
        "deferred": None,
        "draft_accept_pct": round(acc["draft_acc"] * 100.0 / acc["draft"], 1) if acc["draft"] else None,
        "cache_hit_pct": (round(acc["cached"] * 100.0 / (acc["cached"] + acc["processed"]), 1)
                          if acc["cached"] + acc["processed"] else None),
        "kv_used": None,          # /v1/status 不暴露 KV 占用；池大小在 engine_state.kv_pool
        "kv_ratio_pct": None,
        "requests_total": act.get("requests"),
        # "现在"值就是最近一次请求的内部速率（引擎只有这个口径，比窗口平均更细）
        "prefill_now": t.get("prompt_per_second"),
        "decode_now": t.get("predicted_per_second"),
        "is_live": False,
        "measured_at": at,
        "rate_mode": "last_request",
        "prefill_peak": acc["peak_p"],
        "decode_peak": acc["peak_d"],
    }


_hipfire_acc = {"req": None, "prompt": None, "completion": None,
                "p_sum": None, "p_cnt": None, "d_sum": None, "d_cnt": None}


def _metrics_hipfire(txt):
    """hipfire 的 /metrics（prometheus 文本）→ 面板指标。

    /metrics 提供累计请求/提示/生成 token 计数，以及每条请求的
    prefill_tokens_per_second / decode_tokens_per_second 直方图。面板用
    计数器差分补"完成请求采样"（提示/生成 tokens、预填充/输出 tok/s），用
    直方图 sum/count 给累计平均；**峰值只取面板实际采样到的单请求差分值**，
    不使用直方图桶上界。没有 KV/草稿采纳率/延迟原值（卡片对应项显示 —）。"""
    v = {}
    for line in txt.splitlines():
        if not line or line.startswith("#"):
            continue
        name, _, val = line.rpartition(" ")
        try:
            v[name] = float(val)
        except ValueError:
            continue
    g = v.get
    # 换源时清掉共享采样基准/趋势；同一引擎连续采样不再 reset。
    if _metrics_src != "hipfire":
        _settle_rates("hipfire", time.monotonic(), (None, None, None, None))

    req = g("hipfire_requests_total")
    prompt = g("hipfire_prompt_tokens_total")
    completion = g("hipfire_completion_tokens_total")
    cached = g("hipfire_cached_prompt_tokens_total") or 0.0
    p_sum = g("hipfire_prefill_tokens_per_second_sum")
    p_cnt = g("hipfire_prefill_tokens_per_second_count")
    d_sum = g("hipfire_decode_tokens_per_second_sum")
    d_cnt = g("hipfire_decode_tokens_per_second_count")
    p_avg = _div(p_sum, p_cnt)
    d_avg = _div(d_sum, d_cnt)

    acc = _hipfire_acc
    reset = (acc["req"] is not None and (
        req is None or req < acc["req"]
        or (p_sum is not None and acc["p_sum"] is not None and p_sum < acc["p_sum"])
        or (d_sum is not None and acc["d_sum"] is not None and d_sum < acc["d_sum"])))
    if reset:
        _reset_rates()
        acc.update(req=None, prompt=None, completion=None,
                   p_sum=None, p_cnt=None, d_sum=None, d_cnt=None)

    if req is not None and acc["req"] is not None and req > acc["req"]:
        dpt = (prompt - acc["prompt"]
               if prompt is not None and acc["prompt"] is not None else None)
        dct = (completion - acc["completion"]
               if completion is not None and acc["completion"] is not None else None)
        dp_sum = (p_sum - acc["p_sum"]
                  if p_sum is not None and acc["p_sum"] is not None else None)
        dp_cnt = (p_cnt - acc["p_cnt"]
                  if p_cnt is not None and acc["p_cnt"] is not None else None)
        dd_sum = (d_sum - acc["d_sum"]
                  if d_sum is not None and acc["d_sum"] is not None else None)
        dd_cnt = (d_cnt - acc["d_cnt"]
                  if d_cnt is not None and acc["d_cnt"] is not None else None)
        p = _div(dp_sum, dp_cnt) if dp_cnt else None
        d = _div(dd_sum, dd_cnt) if dd_cnt else None
        if p is not None:
            _last_good["prefill"] = p
            _rate_peak["prefill"] = (p if _rate_peak["prefill"] is None
                                     else max(_rate_peak["prefill"], p))
        if d is not None:
            _last_good["decode"] = d
            _rate_peak["decode"] = (d if _rate_peak["decode"] is None
                                    else max(_rate_peak["decode"], d))
        if p is not None or d is not None:
            _rate_history.append({
                "t": int(time.time()), "p": p, "d": d,
                "pt": int(dpt) if dpt and dpt > 0 else None,
                "ct": int(dct) if dct and dct > 0 else None,
            })
            if len(_rate_history) > RATE_HIST_LEN:
                del _rate_history[:-RATE_HIST_LEN]

    acc.update(req=req, prompt=prompt, completion=completion,
               p_sum=p_sum, p_cnt=p_cnt, d_sum=d_sum, d_cnt=d_cnt)
    # 首次采样用累计平均兜底；之后用最后一条完成请求的差分值。
    p_now = (_last_good["prefill"] if _last_good["prefill"] is not None else p_avg)
    d_now = (_last_good["decode"] if _last_good["decode"] is not None else d_avg)
    # 峰值只来自面板实际采到的单请求差分值（_rate_peak 在收到请求时更新）；
    # 不用直方图桶上界——那会把"全部请求 ≤1000"误报成峰值 1000。
    # /metrics 只给 spec_tau summary（平均接受长度），没有草稿/接受 token 计数；
    # 面板在 hipfire 上显示 τ，而不是伪造一个百分比。
    spec_sum = g("hipfire_spec_tau_sum")
    spec_cnt = g("hipfire_spec_tau_count")
    draft_tau = round(spec_sum / spec_cnt, 2) if spec_cnt else None
    return {
        "engine": "hipfire",
        "prefill_avg": p_avg,
        "decode_avg": d_avg,
        "processing": g("hipfire_queue_depth"),
        "deferred": None,
        "draft_accept_pct": None,
        "draft_tau": draft_tau,
        "cache_hit_pct": (round(cached * 100.0 / prompt, 1)
                          if prompt and cached is not None else None),
        "kv_used": None,
        "kv_ratio_pct": None,
        "requests_total": req,
        "prefill_now": p_now,
        "decode_now": d_now,
        "is_live": False,
        "measured_at": int(time.time()),
        "rate_mode": "per_request_histogram",
        "prefill_peak": _rate_peak["prefill"],
        "decode_peak": _rate_peak["decode"],
    }


def _metrics_gsqhalo(txt):
    """gsqhalo 的 /metrics（llama.cpp prometheus 文本）→ 面板指标。

    计数器是上游 llama.cpp 的名字（llamacpp:*）；草稿采纳率来自 spec_decode_*
    三个计数器（MTP 与 ngram-mod 合并计）。这个 build 不暴露 KV 占用与请求
    总数，前端对这两个键显示 —。
    """
    v = {}
    for line in txt.splitlines():
        if not line or line.startswith("#"):
            continue
        name, _, val = line.rpartition(" ")
        try:
            v[name] = float(val)
        except ValueError:
            continue

    g = v.get
    live_p, live_d = _settle_rates(
        "gsqhalo", time.monotonic(),
        (g("llamacpp:prompt_tokens_total"), g("llamacpp:prompt_seconds_total"),
         g("llamacpp:tokens_predicted_total"), g("llamacpp:tokens_predicted_seconds_total")))

    last = _last_rates or {}
    dr = g("llamacpp:spec_decode_num_draft_tokens_total")
    da = g("llamacpp:spec_decode_num_accepted_tokens_total")
    cached = g("llamacpp:prompt_tokens_cached_total")
    proc = g("llamacpp:prompt_tokens_total")

    out = {
        "engine": "gsqhalo",
        "prefill_avg": _div(proc, g("llamacpp:prompt_seconds_total")),
        "decode_avg": _div(g("llamacpp:tokens_predicted_total"),
                           g("llamacpp:tokens_predicted_seconds_total")),
        "processing": g("llamacpp:requests_processing"),
        "deferred": g("llamacpp:requests_deferred"),
        "draft_accept_pct": round(da * 100.0 / dr, 1) if dr and da is not None else None,
        "cache_hit_pct": (round(cached * 100.0 / (cached + proc), 1)
                          if cached is not None and proc is not None and (cached + proc) > 0 else None),
        "kv_used": None,
        "kv_ratio_pct": None,
        "requests_total": None,
    }
    out.update(_now_fields(live_p, live_d, last.get("at")))
    return out


def _metrics_gufo(txt):
    """gufo 的 /metrics → 面板指标（与 _metrics_gsqhalo 同源：它照抄了 llama.cpp
    的 prometheus 名字，所以解析逻辑一致）。差异：gufo 多报 kv_cache_usage_ratio，
    据此给出 kv_ratio_pct；草稿采纳率仍是 spec_decode_* 三个计数器。"""
    v = {}
    for line in txt.splitlines():
        if not line or line.startswith("#"):
            continue
        name, _, val = line.rpartition(" ")
        try:
            v[name] = float(val)
        except ValueError:
            continue

    g = v.get
    live_p, live_d = _settle_rates(
        "gufo", time.monotonic(),
        (g("llamacpp:prompt_tokens_total"), g("llamacpp:prompt_seconds_total"),
         g("llamacpp:tokens_predicted_total"), g("llamacpp:tokens_predicted_seconds_total")))

    last = _last_rates or {}
    dr = g("llamacpp:spec_decode_num_draft_tokens_total")
    da = g("llamacpp:spec_decode_num_accepted_tokens_total")
    cached = g("llamacpp:prompt_tokens_cached_total")
    proc = g("llamacpp:prompt_tokens_total")
    kv = g("llamacpp:kv_cache_usage_ratio")

    out = {
        "engine": "gufo",
        "prefill_avg": _div(proc, g("llamacpp:prompt_seconds_total")),
        "decode_avg": _div(g("llamacpp:tokens_predicted_total"),
                           g("llamacpp:tokens_predicted_seconds_total")),
        "processing": g("llamacpp:requests_processing"),
        "deferred": g("llamacpp:requests_deferred"),
        "draft_accept_pct": round(da * 100.0 / dr, 1) if dr and da is not None else None,
        "cache_hit_pct": (round(cached * 100.0 / (cached + proc), 1)
                          if cached is not None and proc is not None and (cached + proc) > 0 else None),
        "kv_used": None,
        "kv_ratio_pct": round(kv * 100.0, 1) if kv is not None else None,
        "requests_total": None,
    }
    out.update(_now_fields(live_p, live_d, last.get("at")))
    return out


async def _fetch_text(url, total=3):
    try:
        async with ClientSession(timeout=ClientTimeout(total=total)) as s:
            async with _get(s, url) as r:
                if r.status != 200:
                    return None
                return await r.text()
    except Exception:  # noqa: BLE001
        return None


# 8731 是全部引擎共用的端口（互斥，同一时刻只有一个绑定；见上面 ENGINES 的说明）。
# 2026-10-08：判断"这会儿 8731 归谁"用 systemd 单元状态，一次 systemctl 问全部
# 相关单元 + 3 秒缓存（1 Hz 采样下约每 3 秒一次子进程，可忽略不计）。
# 2026-10-17：strata / gsqhalo / hipfire 也统一到 8731，全部纳入。
_UNITS_8731 = (("halogen-docker", "halogen"),
               ("kyojin-qwen", "kyojin"),
               ("strata", "strata"), ("gsqhalo", "gsqhalo"),
               ("hipfire-27b", "hipfire"), ("gufo", "gufo"))
_8731_cache = {"at": 0.0, "engine": None}
_8731_TTL = 3.0


async def engine_on_8731():
    """systemd 眼里这会儿 8731 上答话的是谁：任一引擎 key 或 None。

    只认 active —— activating/deactivating 是引擎切换的过渡期，此时旧进程可能
    还没释放端口、新进程还没绑上，宁可给 None 让调用方跳过本轮指标。引擎没在跑
    （或有人手工起了个不归面板管的野进程）时也是 None。所有引擎现在共用 8731，
    包括 kyojin 的两个变体。"""
    now = time.monotonic()
    if now - _8731_cache["at"] < _8731_TTL:
        return _8731_cache["engine"]
    _, out = await run("systemctl", "--user", "is-active",
                       *[unit for unit, _ in _UNITS_8731], timeout=8)
    who = None
    for (_, owner), line in zip(_UNITS_8731, out.splitlines()):
        if line.strip() == "active":
            who = owner
            break
    _8731_cache.update(at=now, engine=who)
    return who


async def engine_metrics():
    """所有引擎共用 8731（互斥，同一时刻只有一个答话）。先用 systemd 单元状态
    判断当前引擎，再只问它自己的端点：既避免 1 Hz 探测把 404 写进别的引擎日志，
    也避免同端口下按 marker 误判（halogen/kyojin/gsqhalo 的 prometheus 名称相近）。
    返回值的 engine 字段告诉前端数字属于谁；过渡期/未知返回 None。"""
    who = await engine_on_8731()
    if not who:
        return None
    try:
        if who == "halogen":
            txt = await _fetch_text(ENGINES["halogen"]["metrics"])
            if txt and "halogen:" in txt:
                return _metrics_halogen(txt)
        elif who == "kyojin":
            txt = await _fetch_text(ENGINES["kyojin"]["stats"])
            if txt:
                return _metrics_kyojin(txt)
        elif who == "strata":
            txt = await _fetch_text(ENGINES["strata"]["stats"])
            if txt:
                return _metrics_strata(txt)
        elif who == "gsqhalo":
            txt = await _fetch_text(ENGINES["gsqhalo"]["metrics"])
            if txt and "llamacpp:prompt_tokens_total" in txt:
                return _metrics_gsqhalo(txt)
        elif who == "hipfire":
            txt = await _fetch_text(ENGINES["hipfire"]["metrics"])
            if txt:
                return _metrics_hipfire(txt)
        elif who == "gufo":
            txt = await _fetch_text(ENGINES["gufo"]["metrics"])
            # gufo 的 prometheus 文本沿用 llama.cpp 的 llamacpp:* 名字，
            # 另加 gufo_* 扩展；用 counters 名字做归属校验。
            if txt and "llamacpp:prompt_tokens_total" in txt:
                return _metrics_gufo(txt)
    except Exception:  # noqa: BLE001
        return None
    return None


# ------------------------------------------------------------ 系统状态采集

# 最近若干次采样，供前端画 sparkline（快速通道 1 秒一次采样，180 个点约 3 分钟）
HIST_LEN = 180
_history = {"cpu": [], "gpu": [], "npu": [], "mem": [], "gtt": [], "vram": [],
            "disk": [], "net": []}


def push_history(s):
    try:
        gpu = s.get("gpu") or {}
        dio = s.get("disk_io") or {}
        read_mbs, write_mbs = dio.get("read_mbs"), dio.get("write_mbs")
        npu = s.get("npu")
        net = s.get("net") or {}
        prim = next((i for i in (net.get("ifaces") or [])
                     if i.get("name") == net.get("primary")), None)
        rx_mbs = (prim or {}).get("rx_mbs")
        tx_mbs = (prim or {}).get("tx_mbs")
        samples = {
            "cpu": (s.get("cpu") or {}).get("percent"),
            "mem": (s.get("mem") or {}).get("percent"),
            "gpu": gpu.get("busy"),
            # 设备在但没人用 = 0（一条平的线，说明"活着且空闲"）；设备不在才是 None
            "npu": None if not npu or not npu.get("present") else (npu.get("busy") or 0.0),
            "gtt": (gpu.get("gtt") or {}).get("percent"),
            "vram": (gpu.get("vram") or {}).get("percent"),
            "disk": None if read_mbs is None or write_mbs is None else round(read_mbs + write_mbs, 2),
            # 主接口收发合计（与硬盘同口径：只表达"链路有多忙"）
            "net": None if rx_mbs is None or tx_mbs is None else round(rx_mbs + tx_mbs, 2),
        }
        for k, v in samples.items():
            _history[k].append(v)
            if len(_history[k]) > HIST_LEN:
                del _history[k][:-HIST_LEN]
    except Exception:  # noqa: BLE001
        pass


def _rd(path):
    try:
        return Path(path).read_text().strip()
    except Exception:  # noqa: BLE001
        return None


def _ri(path):
    v = _rd(path)
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


_disk_dev_cache = None
_disk_prev = None


def _disk_dev():
    """根分区对应的块设备名（/proc/diskstats 里的名字，如 nvme0n1p2）。"""
    global _disk_dev_cache
    if _disk_dev_cache:
        return _disk_dev_cache
    try:
        with open("/proc/mounts") as f:
            for line in f:
                p = line.split()
                if len(p) >= 3 and p[1] == "/" and p[0].startswith("/dev/"):
                    _disk_dev_cache = os.path.basename(p[0])
                    return _disk_dev_cache
    except OSError:
        pass
    return None


def disk_io():
    """实时磁盘活动：对 /proc/diskstats 的计数器两次采样求差。

    字段按内核规范：p[3] 读完成、p[5] 读扇区、p[7] 写完成、p[9] 写扇区、
    p[12] 设备忙碌毫秒。扇区固定 512 字节（与文件系统块大小无关）。
    """
    global _disk_prev
    dev = _disk_dev()
    if not dev:
        return None
    try:
        with open("/proc/diskstats") as f:
            for line in f:
                p = line.split()
                if len(p) >= 14 and p[2] == dev:
                    cur = (time.monotonic(), int(p[3]), int(p[5]),
                           int(p[7]), int(p[9]), int(p[12]))
                    break
            else:
                return None
    except (OSError, ValueError):
        return None

    prev, _disk_prev = _disk_prev, cur
    if prev is None:
        return {"dev": dev, "read_mbs": None, "write_mbs": None,
                "iops": None, "util_pct": None, "warming": True}
    dt = cur[0] - prev[0]
    if dt <= 0 or any(cur[i] < prev[i] for i in range(1, 6)):
        return None                      # 采样间隔异常或计数器倒退
    return {
        "dev": dev,
        "read_mbs": round((cur[2] - prev[2]) * 512 / dt / 1e6, 2),
        "write_mbs": round((cur[4] - prev[4]) * 512 / dt / 1e6, 2),
        "iops": round((cur[1] - prev[1] + cur[3] - prev[3]) / dt),
        "util_pct": round(min(100.0, (cur[5] - prev[5]) / (dt * 1000) * 100), 1),
    }


# ---------------------------------------------------------------- 网络状态

# 物理网卡的判断：/sys/class/net 里只有真网卡带 device 符号链接，虚拟接口
# （lo / docker0 / veth* / br-*）没有 —— 面板只报告物理链路。
_net_prev = None                      # (monotonic_ts, {iface: (rx_bytes, tx_bytes)})
_wifi_cache = {"at": 0.0, "by": {}}   # iface -> (signal_dbm, ssid, rx_bitrate_mbps)


def _if_ipv4(name):
    """接口的 IPv4 地址。SIOCGIFADDR ioctl，不 spawn 外部命令。"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            r = fcntl.ioctl(s.fileno(), 0x8915, struct.pack("256s", name.encode()[:15]))
        return socket.inet_ntoa(r[20:24])
    except OSError:
        return None


def _netdev_counters():
    """{iface: (rx_bytes, tx_bytes)}，来自 /proc/net/dev。"""
    out = {}
    try:
        with open("/proc/net/dev") as f:
            f.readline()
            f.readline()
            for line in f:
                name, _, rest = line.partition(":")
                p = rest.split()
                if name.strip() and len(p) >= 9:
                    out[name.strip()] = (int(p[0]), int(p[8]))
    except (OSError, ValueError):
        pass
    return out


def _default_route():
    """默认路由的 (接口, 网关)。/proc/net/route 里网关是 16 进制小端。"""
    try:
        with open("/proc/net/route") as f:
            f.readline()
            for line in f:
                p = line.split()
                if len(p) >= 4 and p[1] == "00000000" and int(p[3], 16) & 0x2:
                    return p[0], socket.inet_ntoa(struct.pack("<L", int(p[2], 16)))
    except (OSError, ValueError, struct.error):
        pass
    return None, None


def _parse_iw_link(txt):
    """iw dev <if> link 的输出 -> (signal_dbm, ssid, rx_bitrate_mbps)，全缺返回 None。"""
    sig = ssid = rate = None
    m = re.search(r"^\s*signal:\s*(-?\d+) dBm", txt, re.M)
    if m:
        sig = int(m.group(1))
    m = re.search(r"^\s*SSID:\s*(.+)$", txt, re.M)
    if m:
        ssid = m.group(1).strip()
    m = re.search(r"^\s*rx bitrate:\s*([\d.]+)", txt, re.M)
    if m:
        rate = round(float(m.group(1)))
    if sig is None and ssid is None and rate is None:
        return None
    return (sig, ssid, rate)


def _wifi_link(iface):
    """WiFi 的 SSID / 信号 / 协商速率。按 10 s 缓存地跑一次 `iw`。

    面板的"系统状态全读 /proc 与 /sys、不 spawn 外部命令"在这里开第二处例外
    （第一处是 NPU 的 docker exec 流），原因同类：内核在这台机器上没给 WiFi 的
    /proc、/sys 通路（/proc/net/wireless 空、sysfs 的 wireless/ 目录无文件），
    而信号是 WiFi 唯一的健康读数。10 s 一读 + 3 s 超时 + 失败静默降级（只是少
    一个标签），1 Hz 的采集器不会变成每秒 spawn 一次。"""
    now = time.monotonic()
    if now - _wifi_cache["at"] > 10.0:
        _wifi_cache["at"] = now
        _wifi_cache["by"] = {}
        try:
            names = [n for n in os.listdir("/sys/class/net")
                     if os.path.isdir(f"/sys/class/net/{n}/wireless")]
        except OSError:
            names = []
        for name in names:
            try:
                r = subprocess.run(["iw", "dev", name, "link"],
                                   capture_output=True, text=True, timeout=3)
                if r.returncode == 0:
                    _wifi_cache["by"][name] = _parse_iw_link(r.stdout)
            except (OSError, subprocess.SubprocessError):
                pass
    return _wifi_cache["by"].get(iface)


def net_stat():
    """本机网络状态：物理接口的 up/down、IPv4、实时吞吐（两次采样求差）、
    WiFi 的 SSID/信号/协商速率，以及默认路由接口与网关。"""
    global _net_prev
    counters = _netdev_counters()
    ts = time.monotonic()
    def_if, gateway = _default_route()

    try:
        names = sorted(os.listdir("/sys/class/net"))
    except OSError:
        names = []
    ifaces = []
    for name in names:
        base = f"/sys/class/net/{name}"
        if name == "lo" or not os.path.exists(f"{base}/device"):
            continue
        state = _rd(f"{base}/operstate")
        speed = _ri(f"{base}/speed")
        wifi = os.path.isdir(f"{base}/wireless")
        rec = {
            "name": name,
            "kind": "wifi" if wifi else "eth",
            "operstate": state,
            "up": state == "up" or _rd(f"{base}/carrier") == "1",
            "speed_mbps": speed if speed and speed > 0 else None,
            "ipv4": _if_ipv4(name),
        }
        if wifi:
            link = _wifi_link(name)
            if link:
                rec["signal_dbm"], rec["ssid"], rec["bitrate_rx_mbps"] = link
        rx, tx = counters.get(name, (None, None))
        if rx is not None:
            rec["rx_gb"] = round(rx / 1e9, 2)
            rec["tx_gb"] = round(tx / 1e9, 2)
        ifaces.append(rec)

    warming = True
    if _net_prev and ts - _net_prev[0] > 0:
        dt, prev = ts - _net_prev[0], _net_prev[1]
        warming = False
        for rec in ifaces:
            pr, cur = prev.get(rec["name"]), counters.get(rec["name"])
            if not pr or not cur or cur[0] < pr[0] or cur[1] < pr[1]:
                continue          # 接口新出现或计数器回绕：本轮不给速率
            rec["rx_mbs"] = round((cur[0] - pr[0]) / dt / 1e6, 3)
            rec["tx_mbs"] = round((cur[1] - pr[1]) / dt / 1e6, 3)
    _net_prev = (ts, counters)

    primary = (next((r["name"] for r in ifaces if r["name"] == def_if), None)
               or next((r["name"] for r in ifaces if r["up"]), None)
               or (ifaces[0]["name"] if ifaces else None))
    return {"ifaces": ifaces, "primary": primary, "gateway": gateway,
            "warming": warming}


_cpu_prev = None


def cpu_percent():
    """CPU 总占用率。第一次调用只做基线，返回 0。"""
    global _cpu_prev
    try:
        with open("/proc/stat") as f:
            v = list(map(int, f.readline().split()[1:11]))
    except Exception:  # noqa: BLE001
        return None
    idle = v[3] + v[4]          # idle + iowait
    total = sum(v)
    prev, _cpu_prev = _cpu_prev, (total, idle)
    if prev is None:
        return 0.0
    dt, di = total - prev[0], idle - prev[1]
    if dt <= 0:
        return None
    return round(max(0.0, min(100.0, (dt - di) * 100.0 / dt)), 1)


def cpu_temp():
    """CPU 温度：找 k10temp / zenpower 的 Tctl。"""
    for h in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
        if _rd(f"{h}/name") in ("k10temp", "zenpower", "coretemp"):
            t = _ri(f"{h}/temp1_input")
            if t is not None:
                return round(t / 1000.0, 1)
    return None


def cpu_freq_mhz():
    try:
        cur = []
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("cpu MHz"):
                    cur.append(float(line.split(":")[1]))
        return round(sum(cur) / len(cur)) if cur else None
    except Exception:  # noqa: BLE001
        return None


def gpu_paths():
    """定位 AMD GPU 的 sysfs 目录与 amdgpu hwmon 目录。"""
    dev = hw = None
    for p in sorted(glob.glob("/sys/class/drm/card*/device")):
        if os.path.exists(f"{p}/gpu_busy_percent"):
            dev = p
            break
    if dev:
        for h in sorted(glob.glob(f"{dev}/hwmon/hwmon*")):
            if _rd(f"{h}/name") == "amdgpu":
                hw = h
                break
    return dev, hw


def _clock_mhz(path):
    """pp_dpm_* 里带 * 的那一行是当前档位。"""
    txt = _rd(path)
    if not txt:
        return None
    for line in txt.splitlines():
        if "*" in line:
            for tok in line.replace("*", " ").split():
                if tok.lower().endswith("mhz"):
                    try:
                        return int(float(tok[:-3]))
                    except ValueError:
                        return None
    return None


# ------------------------------------------------------------------ NPU
#
# NPU（amdxdna）不像 GPU 那样有一个 gpu_busy_percent 之类的 sysfs 利用率，也不
# 方便每分钟去 spawn 一次 xrt-smi。它有的是 DRM 的标准账本：每个客户端在
# /proc/<pid>/fdinfo 里被记下"在这个引擎上累计跑了多少纳秒"
# （drm-engine-amdxdna_accel_driver）—— 两次采样求差就是利用率，和 GPU 的
# busy% 同一口径；同用户进程即可读，不需要 root。
#
# 坑：计数器是"每客户端累计"，客户端一退出就随 fd 一起消失。所以基线按
# (pid, fd) 记账，只对两次都在的客户端求差；消失的直接丢弃，否则退出会被
# 算成一次负增量，曲线瞬间掉到 0 以下。
ACCEL_ENGINE_KEY = "drm-engine-amdxdna_accel_driver:"
_accel = {"prev": {}, "t": None}


def npu_clients():
    """NPU 客户端：{(pid, fd): (累计忙碌 ns, drm 记账内存 KiB)}。

    fd 先按符号链接筛（指向 /dev/accel*），只给筛中的读 fdinfo —— 全量读
    一遍 fdinfo 要 26 ms，筛一遍 7 ms。读不到的直接跳过：别的用户的进程
    读不了 fdinfo，不该因此报错（halogen 容器里的 root 进程就属于这种，
    它们的数由 _npu_ctr 那份容器内采样补上，键加 "c" 前缀免得与宿主 pid 撞号）。
    """
    out = {}
    for link in glob.glob("/proc/[0-9]*/fd/*"):
        try:
            if not os.readlink(link).startswith("/dev/accel"):
                continue
            txt = Path(link.replace("/fd/", "/fdinfo/")).read_text()
        except OSError:
            continue
        busy = mem = None
        for line in txt.splitlines():
            if line.startswith(ACCEL_ENGINE_KEY):
                busy = int(line.split(":", 1)[1].split()[0])
            elif line.startswith("drm-total-memory:"):
                mem = int(line.split(":", 1)[1].split()[0])
        if busy is not None:
            out[(link.split("/")[2], os.path.basename(link))] = (busy, mem or 0)
    ctr, ts = _npu_ctr["clients"], _npu_ctr["ts"]
    if ctr and time.monotonic() - ts < 5.0:  # 5 s 内的容器采样才算数
        for (pid, fd), v in ctr.items():
            out[("c" + pid, fd)] = v
    return out


# 容器内的采样循环（root 跑，所以能读 halogen 引擎自己的 fdinfo）。每轮把容器里
# 所有 /dev/accel 客户端打一行 "C <pid> <fd> <busy_ns> <mem_kib>"，以 "E" 收尾；
# 0.5 s 一轮，面板 1 s 采一次刚好跨两轮，不会出现"采到同一轮、差值为 0"的抖动。
NPU_CTR_LOOP = r"""while :; do
for f in /proc/[0-9]*/fdinfo/*; do
  v=$(awk '/^drm-engine-amdxdna_accel_driver:/{b=$2} /^drm-total-memory:/{m=$2} \
END{if(b!="")print b+0, m+0}' "$f" 2>/dev/null)
  [ -n "$v" ] || continue
  p=${f#/proc/}; p=${p%%/*}
  echo "C $p ${f##*/} $v"
done
echo E
sleep 0.5
done"""

_npu_ctr = {"clients": None, "ts": 0.0}


async def sample_npu_container():
    """常驻一条 `docker exec` 流，收容器内 NPU 客户端的累计计数（见 NPU_CTR_LOOP）。

    宿主上读不到 root 容器进程的 /proc/<pid>/fdinfo，所以让容器自己报数；流断掉
    （容器重启 / 停掉 / docker 不可用）就退避 5 s 重连，期间 clients=None，NPU 行
    回落到"设备在、无客户端"。stderr 只在内容变化时记一条，容器停着时不刷日志。
    """
    last_err = None
    while True:
        proc = None
        try:
            proc = await asyncio.create_subprocess_exec(
                DOCKER, "exec", ENGINES["halogen"]["log_container"], "bash", "-c", NPU_CTR_LOOP,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            cur = {}
            async for raw in proc.stdout:
                line = raw.decode(errors="ignore").strip()
                if line == "E":
                    _npu_ctr.update(clients=dict(cur), ts=time.monotonic())
                    cur = {}
                    continue
                parts = line.split()
                if len(parts) == 5 and parts[0] == "C":
                    try:
                        cur[(parts[1], parts[2])] = (int(parts[3]), int(parts[4]))
                    except ValueError:
                        pass
            err = (await proc.stderr.read()).decode(errors="ignore").strip()
            await proc.wait()
            msg = err or f"docker exec 退出（rc={proc.returncode}）"
            if msg != last_err:
                logging.warning("npu 容器采样中断: %s", msg)
                last_err = msg
        except Exception as e:
            if str(e) != last_err:
                logging.warning("npu 容器采样中断: %s", e)
                last_err = str(e)
        finally:
            _npu_ctr.update(clients=None, ts=time.monotonic())
            if proc is not None and proc.returncode is None:
                with suppress(ProcessLookupError):
                    proc.kill()
        await asyncio.sleep(5)


def npu_stat():
    """NPU 概况：利用率 / 客户端内存 / 客户端数 / 固件（无设备时 present=False）。"""
    dev = sorted(glob.glob("/sys/class/accel/accel*"))
    if not dev:
        # 设备不在（驱动没加载）：清掉基线，免得设备回来时拿旧窗口求差
        _accel.update(prev={}, t=None)
        return {"present": False}

    now = time.monotonic()
    clients = npu_clients()
    prev, prev_t = _accel["prev"], _accel["t"]
    busy = None
    if prev_t is not None:
        delta = sum(b - prev[k][0] for k, (b, _) in clients.items() if k in prev)
        # 多客户端同跑时求和可能超 100%（这是"引擎被占用的程度"，不是互斥占用）
        busy = round(min(100.0, max(0.0, delta * 100.0 / ((now - prev_t) * 1e9))), 1)
    _accel.update(prev=clients, t=now)

    return {
        "present": True,
        "busy": busy,
        "mem_mb": round(sum(m for _, m in clients.values()) / 1024.0),
        "clients": len({k[0] for k in clients}),
        "fw": _rd(f"{dev[0]}/device/fw_version"),
    }


def sysinfo():
    info = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        k, _, v = line.partition(":")
        info[k] = int(v.split()[0]) * 1024

    mem_total, mem_avail = info["MemTotal"], info["MemAvailable"]
    mem_used = mem_total - mem_avail

    out = {
        "cpu": {
            "percent": cpu_percent(),
            "temp_c": cpu_temp(),
            "freq_mhz": cpu_freq_mhz(),
            "load1": None,
        },
        "mem": {
            "total_gb": round(mem_total / GIB, 1),
            "used_gb": round(mem_used / GIB, 1),
            "avail_gb": round(mem_avail / GIB, 1),
            "percent": round(mem_used * 100.0 / mem_total, 1) if mem_total else None,
            "swap_used_gb": round((info["SwapTotal"] - info["SwapFree"]) / GIB, 1),
            "swap_total_gb": round(info["SwapTotal"] / GIB, 1),
        },
    }

    try:
        out["cpu"]["load1"] = round(float(Path("/proc/loadavg").read_text().split()[0]), 2)
    except Exception:  # noqa: BLE001
        pass

    # 磁盘：容量与实时活动分开给，前端按"活动类"呈现
    try:
        st = os.statvfs("/")
        total = st.f_blocks * st.f_frsize
        used = total - st.f_bfree * st.f_frsize
        avail = st.f_bavail * st.f_frsize
        out["disk"] = {
            "mount": "/",
            "total_gb": round(total / GIB, 1),
            "used_gb": round(used / GIB, 1),
            "avail_gb": round(avail / GIB, 1),
            "percent": round(used * 100.0 / total, 1) if total else None,
        }
    except Exception:  # noqa: BLE001
        out["disk"] = None
    out["disk_io"] = disk_io()

    # 网络：物理接口的 up/down、IP、实时吞吐、WiFi 信息、默认路由
    out["net"] = net_stat()

    # GPU：利用率 / 温度 / 功耗 / 频率 / 显存 / 共享显存
    dev, hw = gpu_paths()
    if dev:
        vram_total = _ri(f"{dev}/mem_info_vram_total") or 0
        vram_used = _ri(f"{dev}/mem_info_vram_used") or 0
        gtt_total = _ri(f"{dev}/mem_info_gtt_total") or 0
        gtt_used = _ri(f"{dev}/mem_info_gtt_used") or 0
        out["gpu"] = {
            "busy": _ri(f"{dev}/gpu_busy_percent"),
            "temp_c": round(_ri(f"{hw}/temp1_input") / 1000.0, 1) if hw and _ri(f"{hw}/temp1_input") is not None else None,
            "power_w": round(_ri(f"{hw}/power1_average") / 1e6, 1) if hw and _ri(f"{hw}/power1_average") is not None else None,
            "sclk_mhz": _clock_mhz(f"{dev}/pp_dpm_sclk"),
            "fclk_mhz": _clock_mhz(f"{dev}/pp_dpm_fclk"),
            "vram": {
                "used_gb": round(vram_used / GIB, 2),
                "total_gb": round(vram_total / GIB, 2),
                "percent": round(vram_used * 100.0 / vram_total, 1) if vram_total else None,
            },
            "gtt": {
                "used_gb": round(gtt_used / GIB, 2),
                "total_gb": round(gtt_total / GIB, 2),
                "percent": round(gtt_used * 100.0 / gtt_total, 1) if gtt_total else None,
            },
        }
    else:
        out["gpu"] = None

    # NPU：利用率 / 客户端内存 / 客户端数（与 GPU 并列的第二个加速器）
    out["npu"] = npu_stat()

    return out


# 后台唯一采样者；HTTP 客户端只读快照，多个标签页不会扰乱差值基线。
_quick = {"sys": None, "metrics": None, "history": None, "ts": 0,
          "metrics_ts": 0, "metrics_ok": False, "metrics_engine": None}
_active_cache = {"active": None}

async def sample_system():
    data = await asyncio.to_thread(sysinfo)
    push_history(data)
    _quick.update(sys=data, ts=time.time())

async def sample_metrics():
    # 不依赖浏览器是否访问慢接口：模型停止时失败亦不会阻塞系统资源采集。
    # 端口 8731 上答话的引擎由 engine_metrics 自己探测（两者互斥）。
    met = await engine_metrics()
    _quick["metrics_ok"] = met is not None
    _quick["metrics_engine"] = (met or {}).get("engine")
    if met is not None:
        _quick.update(metrics=met, metrics_ts=time.time())

def quick_snapshot():
    return {**_quick, "history": {
        "sys": {k: list(v) for k, v in _history.items()},
        "rates": list(_rate_history)}}

async def sampling_loop(sample):
    while True:
        start = time.monotonic()
        try:
            await sample()
        except Exception:
            logging.exception("control sampler failed: %s", sample.__name__)
        await asyncio.sleep(max(0.05, 1.0 - (time.monotonic() - start)))

async def sampling_context(app):
    tasks = [asyncio.create_task(sampling_loop(sample))
             for sample in (sample_system, sample_metrics)]
    tasks.append(asyncio.create_task(sample_npu_container()))  # 自带重连的常驻流，不走 1 Hz 轮
    yield
    for task in tasks:
        task.cancel()
    for task in tasks:
        with suppress(asyncio.CancelledError):
            await task

async def h_quick(request):
    return web.json_response(quick_snapshot())


async def status(engine: str = DEFAULT_ENGINE, fresh=False):
    active, enabled, uptime, start_key = await unit_state(engine)
    # 模型变体（kyojin 的 GLM/Qwen）：逐个问单元状态；有变体在跑时用它代表整个
    # 引擎（顶栏/按钮/自启开关都按它算），默认单元的登记值只是缺省。
    variants = {}
    vactive = None
    for vk, vcfg in variants_of(engine).items():
        va, ve, vu, vkey = await unit_probe(vcfg["unit"])
        variants[vk] = {"unit": vcfg["unit"], "model": vcfg["model"],
                        "active": va, "enabled": ve, "uptime_s": vu}
        if va in ("active", "activating", "deactivating"):
            vactive = vk
            active, enabled, uptime, start_key = va, ve, vu, vkey
    _active_cache["active"] = active
    if active in ("active", "activating"):
        eng = await engine_state(engine, start_key)
    else:
        eng = {"ready": False, "status": "stopped"}

    # 六个引擎互斥，但切换的过渡期（正在停 A 再起 B）各边状态都要能看见：
    # 前端的确认框与"正在切换"提示依赖 units 与 active_engine。
    units = {engine: {"active": active, "enabled": enabled, "uptime_s": uptime,
                      "variant": vactive}}
    active_engine = engine if active in ("active", "activating", "deactivating") else None
    other_start = {}
    for k in ENGINE_KEYS:
        if k == engine:
            continue
        oa, oe, ou, okey = await unit_state(k)
        other_start[k] = okey
        units[k] = {"active": oa, "enabled": oe, "uptime_s": ou}
        if oa in ("active", "activating", "deactivating"):
            active_engine = k

    snapshot = quick_snapshot()
    sys, met, hist = snapshot["sys"], snapshot["metrics"], snapshot["history"]

    # 选中的引擎没在跑时，正在跑的那个的状态也要给（前端顶栏按 active 引擎描述）。
    # engine_state 按引擎问 8731 上它自己的那个端点。
    engine_states = {engine: eng}
    if active_engine and active_engine != engine:
        engine_states[active_engine] = await engine_state(active_engine, other_start.get(active_engine))

    return {
        "engine_key": engine,
        "unit": units[engine],
        "units": units,
        "variant": vactive,
        "variants": variants,
        # 检查点（halogen）：checkpoint=单元文件里的（下次启动用的）；
        # checkpoint_running=容器 env 里的（正在跑的实例）；二者不同=切换待重启
        "checkpoint": checkpoint_key_now(engine),
        "checkpoints": {k: {"file": c.get("file")}
                        for k, c in checkpoints_of(engine).items()},
        "checkpoint_running": (await checkpoint_running(
            engine, start_key, eng.get("ready"))) if active == "active" else None,
        "active_engine": active_engine,
        "engine": eng,
        "engine_states": engine_states,
        "metrics": met,
        "sys": sys,
        "headroom": (await cache_headroom(engine, start_key, eng.get("ready"))) if active == "active" else None,
        "history": hist,
        "ts": int(time.time()),
    }


# ------------------------------------------------------------ 上游版本检查
#
# 页面打开/刷新时问一次 GitHub：这个仓库只打 tag、不发 release，所以看 tags。
# 匿名 tags 接口限 60 次/小时，远端结果因此缓存 5 分钟——刷新页面不会把它耗光。
# 本地版本取引擎 /health 里的 version.engine，与容器 tag 同源。
UPDATE_REPO = "peonist-ai/halogen-flash-server"
UPDATE_TAGS_API = f"https://api.github.com/repos/{UPDATE_REPO}/tags?per_page=30"
UPDATE_TTL = 300.0        # 远端查询成功后的缓存时长（秒）
UPDATE_FAIL_TTL = 60.0    # 失败后的重试间隔：网络抖一下不该钉住五分钟

_update_remote = {"at": 0.0, "latest": None, "error": None}


def version_key(tag):
    """把 v0.14.0 / 0.13.8 这类标签变成可比较的元组。

    非数字段当 0 处理，够用且不会抛——上游的标签一直是干净的 vX.Y.Z。
    """
    key = []
    for tok in re.split(r"[.\-+_]", str(tag).strip().lstrip("vV")):
        m = re.match(r"\d+", tok)
        key.append(int(m.group()) if m else 0)
    return tuple(key)


async def upstream_latest():
    """GitHub 上最新的 tag，按版本号取最大而不是按接口返回顺序。返回 (标签, 错误)。"""
    now = time.monotonic()
    ttl = UPDATE_FAIL_TTL if _update_remote["error"] else UPDATE_TTL
    if _update_remote["at"] and now - _update_remote["at"] < ttl:
        return _update_remote["latest"], _update_remote["error"]

    latest, error = None, None
    try:
        timeout = ClientTimeout(total=6)
        headers = {"Accept": "application/vnd.github+json",
                   "User-Agent": "halogen-control-panel"}
        async with ClientSession(timeout=timeout) as s:
            async with _get(s, UPDATE_TAGS_API, headers=headers) as r:
                if r.status != 200:
                    error = f"HTTP {r.status}"
                else:
                    tags = [t.get("name") for t in await r.json() if t.get("name")]
                    if tags:
                        latest = max(tags, key=version_key)
                    else:
                        error = "tags 为空"
    except Exception as e:  # noqa: BLE001
        error = f"{type(e).__name__}: {e}"

    _update_remote.update(at=now, latest=latest, error=error)
    return latest, error


# hipfire / gufo 与 halogen 一样按 tag 比较，但仓库不同；各自一份缓存，避免把
# 上游标签混在一起。标签形如 v0.4.2.1 / v0.20.0，version_key 已能处理。
_hipfire_remote = {"at": 0.0, "latest": None, "error": None}
_gufo_remote = {"at": 0.0, "latest": None, "error": None}


async def _latest_tag(repo, cache):
    """GitHub tags 里最大的版本标签（vX.Y[.Z...]），按 repo 各自缓存。"""
    now = time.monotonic()
    ttl = UPDATE_FAIL_TTL if cache["error"] else UPDATE_TTL
    if cache["at"] and now - cache["at"] < ttl:
        return cache["latest"], cache["error"]

    latest, error = None, None
    try:
        headers = {"Accept": "application/vnd.github+json",
                   "User-Agent": "halogen-control-panel"}
        url = f"https://api.github.com/repos/{repo}/tags?per_page=30"
        async with ClientSession(timeout=ClientTimeout(total=6)) as s:
            async with _get(s, url, headers=headers) as r:
                if r.status != 200:
                    error = f"HTTP {r.status}"
                else:
                    # hipfire 的 tags 里混有 safety/win_<hash>_<date> 这类非版本标签；
                    # 只留 vX.Y[.Z...] 的发布标签，否则 version_key 会把它们排到最前。
                    tags = [t.get("name") for t in await r.json()
                            if t.get("name")
                            and re.fullmatch(r"v?\d+\.\d+(?:\.\d+)*", t["name"])]
                    latest = max(tags, key=version_key) if tags else None
                    if not tags:
                        error = "没有版本标签"
    except Exception as e:  # noqa: BLE001
        error = f"{type(e).__name__}: {e}"

    cache.update(at=now, latest=latest, error=error)
    return latest, error


async def hipfire_latest():
    return await _latest_tag("warpfront/hipfire", _hipfire_remote)


async def gufo_latest():
    return await _latest_tag(ENGINES["gufo"]["repo"], _gufo_remote)


# kyojin / strata 是源码构建：用 git 提交标识本地代码，跟 GitHub 默认分支比。
# 一次 compare API 就能同时拿到"落后/领先几个提交"，与 tags 检查共用同一套缓存时长。
_commit_remote = {}       # engine -> 该引擎的缓存（键里带本地 sha）


async def git_head(engine):
    """本地短提交号（如 e9588ac）；读不到返回 None。"""
    rc, out = await run("git", "-C", ENGINES[engine]["dir"],
                        "rev-parse", "--short", "HEAD", timeout=10)
    return out.strip() if rc == 0 and out.strip() else None


async def git_tag(engine):
    """本地 HEAD 的 exact tag（如 v0.4.2.1）；不在 tag 上返回 None。"""
    rc, out = await run("git", "-C", ENGINES[engine]["dir"],
                        "describe", "--tags", "--exact-match", timeout=10)
    return out.strip() if rc == 0 and out.strip() else None


async def git_upstream(engine, local):
    """本地提交 vs 上游默认分支 → (branch, behind, ahead, remote_sha, error)。

    缓存的键里带本地 sha：git pull 换过提交就立刻重查，不受缓存时长影响。"""
    now = time.monotonic()
    r = _commit_remote.setdefault(engine, {"at": 0.0, "local": None, "branch": None,
                                           "behind": None, "ahead": None,
                                           "remote": None, "error": None})
    ttl = UPDATE_FAIL_TTL if r["error"] else UPDATE_TTL
    if r["at"] and r["local"] == local and now - r["at"] < ttl:
        return r["branch"], r["behind"], r["ahead"], r["remote"], r["error"]

    repo = ENGINES[engine]["repo"]
    branch, behind, ahead, remote, error = "main", None, None, None, None
    try:
        rc, out = await run("git", "-C", ENGINES[engine]["dir"],
                            "rev-parse", "--abbrev-ref", "origin/HEAD", timeout=10)
        if rc == 0 and out.strip().startswith("origin/"):
            branch = out.strip().split("/", 1)[1]
        url = f"https://api.github.com/repos/{repo}/compare/{local}...{branch}"
        headers = {"Accept": "application/vnd.github+json",
                   "User-Agent": "halogen-control-panel"}
        async with ClientSession(timeout=ClientTimeout(total=6)) as s:
            async with _get(s, url, headers=headers) as resp:
                if resp.status == 404:
                    error = "本地提交不在上游（可能被 force-push 过）"
                elif resp.status != 200:
                    error = f"HTTP {resp.status}"
                else:
                    d = await resp.json()
                    # compare 是 base...head（这里 base=本地提交、head=上游分支），
                    # 接口的计数是"相对 head"说的：ahead_by = head 比 base 多的提交数
                    # = 本地落后上游多少；behind_by 反之 = 本地多出多少。别搞反。
                    behind = d.get("ahead_by")
                    ahead = d.get("behind_by")
                    commits = d.get("commits") or []
                    remote = commits[-1].get("sha") if commits else local
    except Exception as e:  # noqa: BLE001
        error = f"{type(e).__name__}: {e}"

    r.update(at=now, local=local, branch=branch, behind=behind, ahead=ahead,
             remote=remote, error=error)
    return branch, behind, ahead, remote, error


# ---------------------------------------------------------------- HTTP 层


@web.middleware
async def guard(request, handler):
    """状态变更接口只接受 application/json，以此挡住跨站伪造请求。

    跨站脚本无法在无预检的情况下带 application/json；本服务不发 CORS 头，
    预检必然失败。text/plain 等简单类型能绕过预检，但这里会被拒。
    """
    if request.method == "POST":
        ctype = request.headers.get("Content-Type", "").split(";")[0].strip().lower()
        if ctype != "application/json":
            return web.json_response(
                {"ok": False, "error": msg(request, "ctype")},
                status=415,
            )
    response = await handler(request)
    response.headers["Cache-Control"] = "no-store"
    return response


async def h_status(request):
    return web.json_response(await status(engine_of(request)))


async def h_update(request):
    """本地引擎版本 vs 上游。远端查询有缓存，本地标识每次现取。

    两种口径：halogen 比 GitHub tag（引擎 /health 自带版本号）；
    kyojin / strata / gsqhalo 是源码构建，比的是 git 提交（本地短哈希 + 落后上游几个提交）
    —— strata 虽自报版本号，但标签会落在检出之后，按提交比才准。
    前端按 kind 字段选文案。"""
    engine = engine_of(request)
    if engine in ("hipfire", "gufo"):
        # 与 halogen 同为 tag 口径；hipfire 本地版本来自 checkout 的 exact tag，
        # gufo 来自容器 /props 的 build_info（0.20.0 (0c03748)）。
        eng = await engine_state(engine)
        local = eng.get("version")
        latest, error = (await hipfire_latest() if engine == "hipfire"
                         else await gufo_latest())
        return web.json_response({
            "engine": engine, "kind": "tag", "supported": True,
            "repo": ENGINES[engine]["repo"],
            "local": local,
            "latest": latest,
            "update_available": bool(local and latest
                                     and version_key(latest) > version_key(local)),
            "error": error,
            "checked_at": int(time.time()),
        })
    if engine in ("kyojin", "strata", "gsqhalo"):
        repo = ENGINES[engine]["repo"]
        local = await git_head(engine)
        if not local:
            return web.json_response({
                "engine": engine, "kind": "commit", "supported": False, "repo": repo,
                "local": None, "latest": None, "update_available": False,
                "error": "读不到本地提交（git rev-parse 失败）",
                "checked_at": int(time.time()),
            })
        branch, behind, ahead, remote, error = await git_upstream(engine, local)
        return web.json_response({
            "engine": engine, "kind": "commit", "supported": True, "repo": repo,
            "local": local,
            "latest": (remote or local)[:7],
            "branch": branch, "behind_by": behind, "ahead_by": ahead,
            "update_available": bool(behind),
            "compare_url": f"https://github.com/{repo}/compare/{local}...{branch}",
            "error": error,
            "checked_at": int(time.time()),
        })
    eng = await engine_state(engine)
    local = eng.get("version")
    latest, error = await upstream_latest()
    return web.json_response({
        "engine": engine,
        "kind": "tag",
        "supported": True,
        "repo": UPDATE_REPO,
        "local": local,
        "latest": latest,
        "update_available": bool(local and latest
                                 and version_key(latest) > version_key(local)),
        "error": error,
        "checked_at": int(time.time()),
    })


def _mem_avail_gib():
    """/proc/meminfo 的 MemAvailable，GiB 浮点；读不到回 None。"""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1048576.0
    except (OSError, ValueError, IndexError):
        pass
    return None


async def _wait_launchable(need_gib=105.0, timeout_s=45.0):
    """等 MemAvailable 回到能加载模型的水平再启动，返回实际等待秒数。

    停掉一个引擎后它占的 ~105 GB（strata 约 80 GB）要几秒到几十秒才真正归还；
    紧接着启动时 kyojin 的 serve_glm.sh 100 GiB 门槛会拒绝（exit 3），
    halogen/strata 也会因内存不足而失败或变慢。内存本来就够时立即返回。"""
    t0 = time.monotonic()
    while True:
        avail = _mem_avail_gib()
        if avail is None or avail >= need_gib or time.monotonic() - t0 >= timeout_s:
            return round(time.monotonic() - t0, 1)
        await asyncio.sleep(1.0)


async def h_action(request):
    body = await request.json()
    action = (body or {}).get("action")
    engine = engine_of(request, body)
    if action not in ("start", "stop", "restart"):
        return web.json_response({"ok": False, "error": msg(request, "action")}, status=400)
    variant = variant_key(engine, request, body)
    # 启动/重启没点名变体时跟随"正在跑的那个"：这样对变体引擎点"重启"重启的是
    # 当前模型，而不是永远去起默认那个。
    if action in ("start", "restart") and variant is None:
        variant = await active_variant(engine)
    target = unit_of(engine, variant)

    # 检查点（halogen）：点名了就先把单元文件那一行切过去（原子写+daemon-reload）。
    # 注意改文件对已加载的实例无效——所以点名"与当前不同"的检查点时，即使点的是
    # "启动"也要按重启走（停→起），否则文件切了、跑着的还是旧的。
    cp = checkpoint_key(engine, request, body)
    cp_now = checkpoint_key_now(engine) if checkpoints_of(engine) else None
    cp_switch = bool(cp) and cp != cp_now
    if action in ("start", "restart") and cp_switch:
        changed, err = checkpoint_write(engine, cp)
        if err:
            return web.json_response({"ok": False, "error": err}, status=500)
        if changed:
            await run("systemctl", "--user", "daemon-reload", timeout=20)

    # 互斥：启动/重启 X 之前必须先停掉其余引擎与同一引擎的其它变体（各自要独占本机
    # 大半内存：抢内存时 kyojin 的 serve_*.sh 会自己拒绝启动，halogen/strata 则会
    # 把机器拖垮）。
    stopped_others = []
    stopped_variant = None
    if action in ("start", "restart"):
        for k in ENGINE_KEYS:
            if k == engine:
                continue
            oa, *_ = await unit_state(k)
            if oa in ("active", "activating", "deactivating"):
                rc, out = await run("systemctl", "--user", "stop",
                                    ENGINES[k]["unit"], timeout=90)
                if rc != 0:
                    return web.json_response(
                        {"ok": False, "error": out or msg(request, "rc", rc=rc)}, status=500)
                stopped_others.append(k)
        for vk, vcfg in variants_of(engine).items():
            if vk == variant:
                continue
            oa, *_ = await unit_probe(vcfg["unit"])
            if oa in ("active", "activating", "deactivating"):
                rc, out = await run("systemctl", "--user", "stop", vcfg["unit"], timeout=90)
                if rc != 0:
                    return web.json_response(
                        {"ok": False, "error": out or msg(request, "rc", rc=rc)}, status=500)
                stopped_variant = vcfg["model"]

    if action == "stop":
        # 没点名变体时把该引擎的所有变体单元都停掉（变体共用端口，猜错等于没停）
        units = ([target] if variant is not None
                 else list(dict.fromkeys([ENGINES[engine]["unit"]]
                                         + [c["unit"] for c in variants_of(engine).values()])))
        for unit in units:
            rc, out = await run("systemctl", "--user", "stop", unit, timeout=90)
            if rc != 0:
                return web.json_response(
                    {"ok": False, "error": out or msg(request, "rc", rc=rc)}, status=500)
        return web.json_response({"ok": True, "action": action, "engine": engine,
                                  "variant": variant, "checkpoint": cp_now,
                                  "checkpoint_switched": False,
                                  "stopped_others": [],
                                  "stopped_variant": None, "waited_s": 0.0})

    # 重启拆成"先停、等内存、再起"：systemctl restart 会紧接着启动，而刚退出引擎的
    # GTT 还没归还，启动脚本的内存门槛必然拒绝（2026-10-05 实测过一次 exit 3）。
    # 换检查点的"启动"同理：文件已切，跑着的还是旧实例，必须先停再起。
    if action == "restart" or cp_switch:
        rc, out = await run("systemctl", "--user", "stop", target, timeout=90)
        if rc != 0:
            return web.json_response(
                {"ok": False, "error": out or msg(request, "rc", rc=rc)}, status=500)
    need = 105.0
    if variant:
        need = float(variants_of(engine)[variant].get("min_avail_gib") or need)
    else:
        need = float(ENGINES[engine].get("min_avail_gib") or need)
    waited = await _wait_launchable(need) if action in ("start", "restart") else 0.0

    # 先清"启动限流"计数：单元配了 StartLimitIntervalSec=600 / StartLimitBurst=3
    # （防崩溃循环），但频繁切换（换检查点/换变体各算一次启动）会把配额用光，
    # 第 4 次启动被 systemd 拒绝并把单元标 failed——那是保护机制误伤正常运维。
    # reset-failed 幂等、对健康单元无副作用；systemd 自己的 Restart=on-failure
    # 重启不经过这里，崩溃循环保护照旧。
    await run("systemctl", "--user", "reset-failed", target, timeout=20)

    rc, out = await run("systemctl", "--user",
                        "start" if action == "restart" else action,
                        "--no-block", target, timeout=20)
    if rc != 0:
        return web.json_response({"ok": False, "error": out or msg(request, "rc", rc=rc)}, status=500)
    return web.json_response({"ok": True, "action": action, "engine": engine,
                              "variant": variant,
                              "checkpoint": (cp or cp_now),
                              "checkpoint_switched": cp_switch,
                              "stopped_others": stopped_others,
                              "stopped_variant": stopped_variant, "waited_s": waited})


async def h_boot(request):
    body = await request.json()
    engine = engine_of(request, body)
    verb = "enable" if (body or {}).get("enabled") else "disable"
    variant = variant_key(engine, request, body)
    # 没点名变体时跟随"正在跑的那个"（和按钮一致），都没有就跑默认单元。
    if variant is None:
        variant = await active_variant(engine)
    target = unit_of(engine, variant)
    # 多个引擎都开机自启会在开机时互抢内存；启用一个就自动取消其余（含同引擎的
    # 其它变体 —— 变体之间同样抢内存、抢 8731）。
    disabled_others = []
    disabled_variant = None
    if verb == "enable":
        for k in ENGINE_KEYS:
            if k == engine:
                continue
            # 只有对方本来就在自启时才报"已取消"，免得提示里出现没发生的事
            oa, oe, *_ = await unit_state(k)
            if oe != "enabled":
                continue
            rc, _ = await run("systemctl", "--user", "disable", ENGINES[k]["unit"], timeout=20)
            if rc == 0:
                disabled_others.append(k)
        for vk, vcfg in variants_of(engine).items():
            if vk == variant:
                continue
            if (await unit_probe(vcfg["unit"]))[1] == "enabled":
                rc, _ = await run("systemctl", "--user", "disable", vcfg["unit"], timeout=20)
                if rc == 0:
                    disabled_variant = vcfg["model"]
    units = ([target] if variant is not None or not variants_of(engine)
             else list(dict.fromkeys([ENGINES[engine]["unit"]]
                                     + [c["unit"] for c in variants_of(engine).values()])))
    for unit in units:
        rc, out = await run("systemctl", "--user", verb, unit, timeout=20)
        if rc != 0:
            return web.json_response({"ok": False, "error": out or msg(request, "rc", rc=rc)}, status=500)
    return web.json_response({"ok": True, "enabled": verb == "enable", "engine": engine,
                              "variant": variant, "disabled_others": disabled_others,
                              "disabled_variant": disabled_variant})


async def _engine_logs(engine, tail=3000, timestamps=False, variant=None):
    """按引擎/变体取日志：halogen 在 snap docker 的 json-file（docker logs），
    gsqhalo 在 rootless podman（podman logs，形态与 docker logs 一致），
    kyojin/strata 是 systemd 用户单元（journalctl，-o cat 与 docker logs 的纯文本形态一致）。"""
    cfg = ENGINES[engine]
    if cfg["log_kind"] in ("docker", "podman"):
        cli = DOCKER if cfg["log_kind"] == "docker" else PODMAN
        args = [cli, "logs"]
        if timestamps:
            args.append("--timestamps")
        args += ["--tail", str(tail), cfg["log_container"]]
        rc, out = await run(*args, timeout=25)
        # 引擎没在跑时 docker 回 "No such container"、podman 回 "no container with name
        # ... found"；看日志的人需要的是"这个引擎没有日志"，不是一句 CLI 报错，
        # 换成一行说明（导出日志时亦然）。
        if rc != 0 and "no such container" in out.lower():
            return 0, f"({cfg['label']} 未运行：容器 {cfg['log_container']} 不存在，启动后这里会有日志)"
        return rc, out
    log_unit = cfg["log_unit"]
    if variant in variants_of(engine):
        log_unit = variants_of(engine)[variant].get("log_unit") or log_unit
    args = ["journalctl", "--user", "-u", log_unit, "--no-pager",
            "-n", str(tail), "-o", "short-iso" if timestamps else "cat"]
    return await run(*args, timeout=25)


async def log_variant(engine, request):
    """日志按哪个单元取：请求点名的变体 → 它；没点名但变体在跑 → 在跑的那个。"""
    variant = variant_key(engine, request)
    if variant is None and variants_of(engine):
        variant = await active_variant(engine)
    return variant


async def h_logs(request):
    engine = engine_of(request)
    variant = await log_variant(engine, request)
    # 面板自身的探测占了日志的九成以上，所以要先把窗口开得足够大，
    # 过滤之后才剩得下有意义的历史。
    _, out = await _engine_logs(engine, variant=variant)
    # Docker 版每行只记一次（podman 时代 container+conmon 双记的去重逻辑已不需要，
    # 留着反而会吞掉真实重复行）。
    lines, hidden = [], 0
    show_all = request.query.get("all") in ("1", "true")
    for ln in out.splitlines():
        if not ln:
            continue
        # 面板自己每 3 秒探测 /health 与 /metrics，其访问日志会淹没真正有用的行。
        # 默认隐藏；需要看全量时用 /api/logs?all=1。
        if not show_all and _PROBE_NOISE.search(ln):
            hidden += 1
            continue
        lines.append(ln)
    return web.json_response({"ok": True, "lines": lines[-120:], "hidden": hidden})


async def h_export_status(request):
    """状态快照下载，便于留证或贴到 issue 里。现算一份完整的，不依赖缓存。"""
    engine = engine_of(request)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return web.json_response(await status(engine, fresh=True), headers={
        "Content-Disposition": f'attachment; filename="{engine}-status-{stamp}.json"'})


async def h_export_logs(request):
    engine = engine_of(request)
    variant = await log_variant(engine, request)
    _, out = await _engine_logs(engine, tail=5000, timestamps=True, variant=variant)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return web.Response(text=out, content_type="text/plain", charset="utf-8",
                        headers={"Content-Disposition":
                                 f'attachment; filename="{engine}-logs-{stamp}.log"'})


async def h_index(request):
    return web.Response(text=page_html(), content_type="text/html", charset="utf-8")


# ---------------------------------------------------------------- 页面模板

PAGE_FILE = Path(__file__).with_name("halogen-control.html")
_page_cache = (None, "")


def page_html() -> str:
    """从同目录的 html 读页面；按 mtime 缓存，改完刷新浏览器即可生效。"""
    global _page_cache
    try:
        mt = PAGE_FILE.stat().st_mtime
        if mt != _page_cache[0]:
            _page_cache = (mt, PAGE_FILE.read_text(encoding="utf-8"))
    except OSError as e:
        return f"<h1>页面文件读取失败 / Failed to read the page file</h1><pre>{e}</pre>"
    return _page_cache[1]


app = web.Application(middlewares=[guard])
app.cleanup_ctx.append(sampling_context)
app.router.add_get("/", h_index)
app.router.add_get("/api/status", h_status)
app.router.add_get("/api/quick", h_quick)
app.router.add_get("/api/update", h_update)
app.router.add_get("/api/logs", h_logs)
app.router.add_get("/api/export/status", h_export_status)
app.router.add_get("/api/export/logs", h_export_logs)
app.router.add_post("/api/action", h_action)
app.router.add_post("/api/boot", h_boot)





if __name__ == "__main__":
    print(f"halogen 控制台监听 {LISTEN_HOST}:{LISTEN_PORT}（无访问口令）", flush=True)
    web.run_app(app, host=LISTEN_HOST, port=LISTEN_PORT, access_log=None)
