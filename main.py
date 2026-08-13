import asyncio
import hashlib
import json
import re
import time
from typing import Any, Optional, Dict, Tuple, List

import httpx
from core.plugin import BasePlugin, logger, register_tool as tool, on, Priority
from core.chat.message_utils import KiraMessageEvent, KiraMessageBatchEvent
from core.chat.message_elements import Text, At
from core.chat import MessageChain


def _protect_segments(text: str, protect_idents: bool = False) -> Tuple[str, List[str]]:
    """保护代码块 / 行内代码 / URL（可选：owner/repo 与含下划线标识符）。"""
    buckets: List[str] = []

    def _save(m):
        buckets.append(m.group(0))
        return f"\x00KEEP{len(buckets)-1}\x00"

    text = re.sub(r"```[\s\S]*?```", _save, text)
    text = re.sub(r"`[^`\n]+`", _save, text)
    text = re.sub(r"https?://[^\s<>\"']+", _save, text)
    if protect_idents:
        # owner/repo、路径、snake_case 等：禁止被斜体/粗体规则误伤
        text = re.sub(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_./-]+", _save, text)
        text = re.sub(r"\b[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+\b", _save, text)
    return text, buckets


def _restore_segments(text: str, buckets: List[str]) -> str:
    for i, raw in enumerate(buckets):
        text = text.replace(f"\x00KEEP{i}\x00", raw)
    return text


# Unicode 数学字母（仅映射拉丁/数字；中文等原样保留）
_BOLD_MAP = str.maketrans(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
    "𝗔𝗕𝗖𝗗𝗘𝗙𝗚𝗛𝗜𝗝𝗞𝗟𝗠𝗡𝗢𝗣𝗤𝗥𝗦𝗧𝗨𝗩𝗪𝗫𝗬𝗭𝗮𝗯𝗰𝗱𝗲𝗳𝗴𝗵𝗶𝗷𝗸𝗹𝗺𝗻𝗼𝗽𝗾𝗿𝘀𝘁𝘂𝘃𝘄𝘅𝘆𝘇𝟬𝟭𝟮𝟯𝟰𝟱𝟲𝟳𝟴𝟵",
)
_ITALIC_MAP = str.maketrans(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
    "𝘈𝘉𝘊𝘋𝘌𝘍𝘎𝘏𝘐𝘑𝘒𝘓𝘔𝘕𝘖𝘗𝘘𝘙𝘚𝘛𝘜𝘝𝘞𝘟𝘠𝘡𝘢𝘣𝘤𝘥𝘦𝘧𝘨𝘩𝘪𝘫𝘬𝘭𝘮𝘯𝘰𝘱𝘲𝘳𝘴𝘵𝘶𝘷𝘸𝘹𝘺𝘻",
)
_MONO_MAP = str.maketrans(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
    "𝙰𝙱𝙲𝙳𝙴𝙵𝙶𝙷𝙸𝙹𝙺𝙻𝙼𝙽𝙾𝙿𝚀𝚁𝚂𝚃𝚄𝚅𝚆𝚇𝚈𝚉𝚊𝚋𝚌𝚍𝚎𝚏𝚐𝚑𝚒𝚓𝚔𝚕𝚖𝚗𝚘𝚙𝚚𝚛𝚜𝚝𝚞𝚟𝚠𝚡𝚢𝚣𝟶𝟷𝟸𝟹𝟺𝟻𝟼𝟽𝟾𝟿",
)


def _to_bold(s: str) -> str:
    return s.translate(_BOLD_MAP)


def _to_italic(s: str) -> str:
    return s.translate(_ITALIC_MAP)


def _to_monospace(s: str) -> str:
    return s.translate(_MONO_MAP)


def sanitize_qq_text(text: str) -> str:
    """仅剥离 Markdown 语法标记，绝不改写实际字符（尤其是下划线 _）。"""
    if not text:
        return text

    text, buckets = _protect_segments(text, protect_idents=False)

    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text, flags=re.DOTALL)
    # 不处理 __bold__，避免误伤下划线
    text = re.sub(r"(?<!\*)\*(?!\*)([^*\n]+?)(?<!\*)\*(?!\*)", r"\1", text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^>\s?", "", text, flags=re.MULTILINE)

    text = _restore_segments(text, buckets)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _has_cjk(s: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]", s))


def _to_fullwidth_alnum(s: str) -> str:
    """拉丁字母/数字 → 全角（BMP，QQ/NapCat 兼容，比数学粗体更稳）。"""
    out = []
    for ch in s:
        o = ord(ch)
        if 0x41 <= o <= 0x5A:
            out.append(chr(0xFF21 + (o - 0x41)))
        elif 0x61 <= o <= 0x7A:
            out.append(chr(0xFF41 + (o - 0x61)))
        elif 0x30 <= o <= 0x39:
            out.append(chr(0xFF10 + (o - 0x30)))
        else:
            out.append(ch)
    return "".join(out)


def _style_bold(s: str) -> str:
    """粗体：英文全角 + 中文【】，替代 Markdown **。"""
    s = s.strip()
    if not s:
        return s
    if _has_cjk(s):
        return f"【{_to_fullwidth_alnum(s)}】"
    return _to_fullwidth_alnum(s)


def _style_italic(s: str) -> str:
    """斜体：统一「」，替代 Markdown *。"""
    s = s.strip()
    if not s:
        return s
    return f"「{s}」"


def stylize_qq_text(text: str) -> str:
    """Markdown 强调 → 美观全角/【】/「」（不用数学字母，避免 NapCat 伪造转发失败）。"""
    if not text:
        return text

    text, buckets = _protect_segments(text, protect_idents=True)

    text = re.sub(
        r"^#{1,6}\s*(.+?)\s*$",
        lambda m: _style_bold(m.group(1)),
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"\*\*(.+?)\*\*",
        lambda m: _style_bold(m.group(1)),
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"(?<!\*)\*(?!\*)([^*\n]+?)(?<!\*)\*(?!\*)",
        lambda m: _style_italic(m.group(1)),
        text,
    )

    text = _restore_segments(text, buckets)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_empty_template(tmpl: Any) -> bool:
    """空字符串 / 空对象 / 空 lines → 视为未配置。"""
    if tmpl is None:
        return True
    if isinstance(tmpl, str):
        return not tmpl.strip()
    if isinstance(tmpl, dict):
        if not tmpl:
            return True
        if "lines" in tmpl:
            lines = tmpl.get("lines")
            return not lines
        for k in ("content", "text", "value", "body"):
            if k in tmpl:
                return not str(tmpl.get(k) or "").strip()
        return False
    if isinstance(tmpl, (list, tuple)):
        return len(tmpl) == 0
    return False


# ============================================================
# DeepWiki MCP 客户端（复用原有逻辑，稍作增强）
# ============================================================

class _RetryableError(Exception):
    """标记可重试的瞬时故障（HTTP 429/5xx、网络抖动、服务端过载错误文本）。"""
    pass


# 瞬时 HTTP 状态码：重试有意义
_TRANSIENT_HTTP_CODES = {429, 500, 502, 503, 504}


# 服务端错误文本中的瞬时故障特征（如 "Server error '503 Service Temporarily Unavailable'..."）
_TRANSIENT_TEXT_RE = re.compile(
    r"(?<!\d)(?:50[0-4]|429)(?!\d)"
    r"|temporarily unavailable|service unavailable|server error"
    r"|bad gateway|overloaded|timed?\s*out|connection (?:reset|refused|closed)"
    r"|error processing question",
    re.IGNORECASE,
)


def _is_transient_text(text: str) -> bool:
    """判断服务端返回文本是否属于瞬时故障（适合重试）。"""
    return bool(_TRANSIENT_TEXT_RE.search(text or ""))


class DeepWikiClient:
    def __init__(self, mcp_url: str, protocol_version: str, timeout: float, max_retries: int = 3):
        self.mcp_url = mcp_url
        self.protocol_version = protocol_version
        self.timeout = timeout
        # 瞬时故障（429/5xx/网络抖动/服务端过载）最大重试次数；退避 1s/2s/4s...
        self.max_retries = max_retries
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def _call_mcp_tool(self, tool_name: str, arguments: dict) -> str:
        """通用 MCP 工具调用，支持 ask_question / read_wiki_structure / read_wiki_contents

        对瞬时故障自动指数退避重试（最多 max_retries 次重试）：
        HTTP 429/5xx、网络异常/超时、服务端错误文本（如 "Server error '503...'"）。
        全部重试仍失败才返回错误串，上层（LLM 工具 / /dw 命令）完全无感。
        """
        client = await self._get_client()
        headers = {
            "Content-Type": "application/json",
            "MCP-Protocol-Version": self.protocol_version,
            "Accept": "application/json, text/event-stream",
        }
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments,
            },
        }
        last_err = "unknown error"
        for attempt in range(self.max_retries + 1):
            try:
                resp = await client.post(self.mcp_url, json=payload, headers=headers)
                # 瞬时 HTTP 状态码：直接重试
                if resp.status_code in _TRANSIENT_HTTP_CODES:
                    raise _RetryableError(f"HTTP {resp.status_code}")
                text = resp.text
                if not text:
                    raise _RetryableError("Empty response from DeepWiki MCP")

                full_answer = []
                for line in text.splitlines():
                    line = line.strip()
                    if line.startswith("data: "):
                        json_str = line[6:]
                        try:
                            data = json.loads(json_str)
                            if "result" in data and "content" in data["result"]:
                                content = data["result"]["content"]
                                if isinstance(content, list) and len(content) > 0:
                                    full_answer.append(content[0].get("text", ""))
                                elif isinstance(content, dict):
                                    full_answer.append(content.get("text", ""))
                            elif "error" in data:
                                msg = f"MCP error: {data['error'].get('message', 'Unknown')}"
                                if _is_transient_text(msg):
                                    raise _RetryableError(msg)
                                return msg
                        except json.JSONDecodeError:
                            continue

                if full_answer:
                    answer = "\n".join(full_answer)
                    # 服务端把上游错误当正常文本返回（如 "Error processing question: Server error '503...'"）
                    if _is_transient_text(answer):
                        raise _RetryableError(answer)
                    return answer

                # fallback: 直接 JSON
                data = json.loads(text)
                if "result" in data and "content" in data["result"]:
                    content = data["result"]["content"]
                    if isinstance(content, list) and len(content) > 0:
                        answer = content[0].get("text", "")
                        if _is_transient_text(answer):
                            raise _RetryableError(answer)
                        return answer
                    elif isinstance(content, dict):
                        answer = content.get("text", "")
                        if _is_transient_text(answer):
                            raise _RetryableError(answer)
                        return answer
                elif "error" in data:
                    msg = f"MCP error: {data['error'].get('message', 'Unknown')}"
                    if _is_transient_text(msg):
                        raise _RetryableError(msg)
                    return msg
                else:
                    return f"Unexpected response: {str(data)[:200]}"

            except _RetryableError as e:
                last_err = str(e)
                if attempt >= self.max_retries:
                    break
                logger.warning(
                    f"DeepWiki MCP transient error (attempt {attempt + 1}/{self.max_retries + 1}): {e}"
                )
                await asyncio.sleep(2 ** attempt)
            except (httpx.HTTPStatusError, httpx.TransportError, httpx.TimeoutException, asyncio.TimeoutError) as e:
                last_err = str(e)
                if attempt >= self.max_retries:
                    break
                logger.warning(
                    f"DeepWiki MCP network error (attempt {attempt + 1}/{self.max_retries + 1}): {e}"
                )
                await asyncio.sleep(2 ** attempt)
            except Exception as e:
                logger.error(f"DeepWiki MCP call failed: {e}")
                return f"DeepWiki request failed: {str(e)}"

        logger.error(f"DeepWiki MCP call failed after {self.max_retries + 1} attempts: {last_err}")
        return f"DeepWiki request failed: {last_err}"

    async def ask_question(self, repo: str, question: str) -> str:
        return await self._call_mcp_tool(
            "ask_question",
            {"repoName": repo, "question": question}
        )

    async def read_wiki_structure(self, repo: str) -> str:
        return await self._call_mcp_tool(
            "read_wiki_structure",
            {"repoName": repo}
        )

    async def read_wiki_contents(self, repo: str, topic: str = "") -> str:
        args = {"repoName": repo}
        if topic:
            args["topic"] = topic
        return await self._call_mcp_tool("read_wiki_contents", args)

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None
