"""Consultas somente de leitura ao backend Node do portfólio."""

import os
import unicodedata
from contextvars import ContextVar
from contextlib import asynccontextmanager

import httpx
from langchain_core.tools import ToolException

_cache = ContextVar("backend_request_cache", default=None)


@asynccontextmanager
async def request_cache():
    token = _cache.set({})
    try:
        yield
    finally:
        _cache.reset(token)


def get(path: str, expected_type: type):
    cache = _cache.get()
    key = (path, expected_type)
    if cache is not None and key in cache:
        return cache[key]
    base_url = os.getenv("PORTFOLIO_BACKEND_URL", "http://localhost:3000").rstrip("/")
    try:
        response = httpx.get(f"{base_url}{path}", timeout=10.0)
        response.raise_for_status()
        data = response.json()
    except httpx.TimeoutException as exc:
        raise ToolException("O backend demorou para responder. Tente novamente mais tarde.") from exc
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            raise ToolException("Registro não encontrado no backend.") from exc
        raise ToolException("O backend não conseguiu concluir a consulta.") from exc
    except httpx.RequestError as exc:
        raise ToolException("Não foi possível conectar ao backend do portfólio.") from exc
    except ValueError as exc:
        raise ToolException("O backend retornou uma resposta inválida.") from exc
    if not isinstance(data, expected_type) or (
        expected_type is list and any(not isinstance(item, dict) for item in data)
    ):
        raise ToolException("O backend retornou um formato inesperado.")
    if cache is not None:
        cache[key] = data
    return data


def owner() -> dict:
    # Mesmo proprietário definido em users.service.js; nunca usar o primeiro usuário.
    user = next((item for item in get("/user/get", list) if item.get("id") == 1), None)
    if user is None:
        raise ToolException("O perfil do proprietário ainda não foi cadastrado.")
    return user


def normalize(value: str | None) -> str:
    text = unicodedata.normalize("NFKD", (value or "").strip().casefold())
    return "".join(char for char in text if not unicodedata.combining(char))
