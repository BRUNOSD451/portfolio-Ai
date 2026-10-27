"""Gravação das mensagens dos visitantes na API Node do portfólio."""

import asyncio
import logging
import os

import httpx

logger = logging.getLogger("portfolio_ai.messages")

MAX_CONTENT_LENGTH = 10000
MAX_SESSION_LENGTH = 100
_pending = set()
_missing_config_logged = False


def save_message(content: str, session: str | None = None) -> None:
    """Agenda a gravação sem bloquear a resposta; falhas são apenas registradas."""
    global _missing_config_logged
    base_url = os.getenv("PORTFOLIO_API_URL", "").rstrip("/")
    api_key = os.getenv("MESSAGES_API_KEY")
    if not base_url or not api_key:
        if not _missing_config_logged:
            _missing_config_logged = True
            logger.warning("message_save_disabled reason=missing_config")
        return
    if session is not None:
        session = session[:MAX_SESSION_LENGTH]
    task = asyncio.create_task(_post(base_url, api_key, content[:MAX_CONTENT_LENGTH], session))
    # Mantém a referência até o fim para a tarefa não ser coletada.
    _pending.add(task)
    task.add_done_callback(_pending.discard)


async def _post(base_url: str, api_key: str, content: str, session: str | None) -> None:
    # Não registra o conteúdo nem a chave. O timeout cobre o Render free acordando.
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=60.0)) as client:
            response = await client.post(
                f"{base_url}/messages/create",
                json={"conteudo": content, "sessao": session},
                headers={"X-Api-Key": api_key},
            )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        logger.warning("message_save_failed status=%s", exc.response.status_code)
    except Exception as exc:
        logger.warning("message_save_failed error_type=%s", type(exc).__name__)
