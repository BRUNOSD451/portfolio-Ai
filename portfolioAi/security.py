"""Autorização antes do multipart e limites locais de consumo da API."""

import json
import asyncio
import os
import re
import sqlite3
import time
from pathlib import Path
from contextlib import closing

import httpx
from starlette.responses import JSONResponse
from pydantic import ValidationError
from chat_schema import ChatRequest


async def verify_admin(token: str) -> bool:
    # A sessão é opaca. Somente o Node decide se ela existe, expirou ou foi revogada.
    base = os.getenv("PORTFOLIO_BACKEND_URL", "http://localhost:3000").rstrip("/")
    async with httpx.AsyncClient(timeout=5.0, follow_redirects=False) as client:
        response = await client.get(f"{base}/auth/session", headers={"Authorization": f"Bearer {token}"})
    if response.status_code == 401:
        return False
    response.raise_for_status()
    data = response.json()
    return data.get("admin", {}).get("role") == "admin" and data.get("expiresAt", 0) > time.time() * 1000


class SecurityMiddleware:
    def __init__(self, app, verify=verify_admin, budget_path=None, chat_daily_limit=None):
        self.app = app
        self.verify = verify
        self.budget_path = Path(budget_path or os.getenv(
            "AI_SECURITY_DATABASE_PATH", str(Path(__file__).parent / "data" / "security.sqlite3")
        ))
        self.chat_daily_limit = int(chat_daily_limit or os.getenv("CHAT_DAILY_LIMIT", "100"))
        if not 1 <= self.chat_daily_limit <= 10000:
            raise ValueError("CHAT_DAILY_LIMIT deve estar entre 1 e 10000.")
        self.windows = {}
        self.active = {"chat": 0, "knowledge": 0}

    def allow(self, key, limit, seconds):
        now = time.monotonic()
        # Limita também a memória de contadores, mesmo com muitos IPs distintos.
        if len(self.windows) >= 4096:
            self.windows = {k: v for k, v in self.windows.items() if v[0] > now}
            if len(self.windows) >= 4096 and key not in self.windows:
                return False
        end, count = self.windows.get(key, (now + seconds, 0))
        if end <= now:
            end, count = now + seconds, 0
        if count >= limit:
            return False
        self.windows[key] = (end, count + 1)
        return True

    def reserve_daily(self, action, limit):
        self.budget_path.parent.mkdir(parents=True, exist_ok=True)
        day = time.strftime("%Y-%m-%d", time.gmtime())
        with closing(sqlite3.connect(self.budget_path, timeout=2)) as connection, connection:
            connection.execute("CREATE TABLE IF NOT EXISTS budget (day TEXT, action TEXT, count INTEGER, PRIMARY KEY(day, action))")
            connection.execute("DELETE FROM budget WHERE day < ?", (day,))
            result = connection.execute(
                "INSERT INTO budget VALUES (?, ?, 1) ON CONFLICT(day, action) DO UPDATE SET count=count+1 WHERE count<?",
                (day, action, limit),
            )
            return result.rowcount == 1

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] == "OPTIONS":
            return await self.app(scope, receive, send)
        path = scope["path"].rstrip("/")
        if path not in ("/chat", "/knowledge"):
            return await self.app(scope, receive, send)

        async def reject(status, message):
            headers = {"Cache-Control": "no-store"}
            if status == 429:
                headers["Retry-After"] = "60"
            await JSONResponse({"detail": message}, status_code=status, headers=headers)(scope, receive, send)

        ip = (scope.get("client") or ("unknown",))[0]
        if not self.allow("global", 120, 60) or not self.allow((ip, path), 10 if path == "/chat" else 20, 600):
            return await reject(429, "Muitas consultas. Aguarde antes de tentar novamente.")
        if path == "/knowledge":
            headers = dict(scope.get("headers", []))
            authorization = headers.get(b"authorization", b"").decode("latin1")
            match = re.fullmatch(r"Bearer ([A-Za-z0-9_-]{43})", authorization)
            if not match:
                return await reject(401, "Entre no app para gerenciar a base de conhecimento.")
            try:
                valid = await self.verify(match[1])
            except (httpx.HTTPError, ValueError, TypeError, AttributeError):
                return await reject(503, "Não foi possível validar a sessão. Tente novamente.")
            if not valid:
                return await reject(401, "Sessão inválida ou expirada. Entre novamente.")
        if scope["method"] != "POST":
            return await self.app(scope, receive, send)

        action = "chat" if path == "/chat" else "knowledge"
        if self.active[action] >= (2 if action == "chat" else 1):
            return await reject(429, "Há consultas em andamento. Aguarde.")
        self.active[action] += 1
        try:
            # Limite durante a leitura: também funciona sem Content-Length ou com chunked.
            maximum = 192 * 1024 if action == "chat" else 272 * 1024
            body = bytearray()
            try:
                async with asyncio.timeout(20):
                    while True:
                        message = await receive()
                        if message["type"] == "http.disconnect":
                            return
                        body.extend(message.get("body", b""))
                        if len(body) > maximum:
                            return await reject(413, "Corpo da requisição muito grande.")
                        if not message.get("more_body", False):
                            break
            except TimeoutError:
                return await reject(408, "O envio demorou demais. Tente novamente.")
            # JSON inválido não pode esgotar a cota diária das chamadas ao provedor.
            if action == "chat":
                try:
                    value = json.loads(body)
                    ChatRequest.model_validate(value)
                except ValidationError:
                    return await reject(422, "Pergunta ou histórico inválido ou acima dos limites.")
                except (ValueError, UnicodeDecodeError):
                    return await reject(400, "JSON inválido.")
            try:
                if not self.reserve_daily(action, self.chat_daily_limit if action == "chat" else 10):
                    return await reject(429, "Limite diário de consultas atingido. Tente novamente amanhã.")
            except (OSError, sqlite3.Error):
                return await reject(503, "Não foi possível verificar a cota de consultas.")

            delivered = False
            async def replay():
                nonlocal delivered
                if not delivered:
                    delivered = True
                    return {"type": "http.request", "body": bytes(body), "more_body": False}
                return await receive()
            await self.app(scope, replay, send)
        finally:
            self.active[action] -= 1
