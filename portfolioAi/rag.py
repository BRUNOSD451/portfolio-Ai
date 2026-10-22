"""Indexação de TXT e busca semântica persistida em SQLite."""

import hashlib
import json
import math
import os
import re
import sqlite3
import ssl
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

import httpx
import truststore

MAX_BYTES = 256 * 1024
INGEST_LOCK = Lock()


class RagError(Exception):
    def __init__(self, message: str, status: int = 502):
        super().__init__(message)
        self.status = status


def _connect():
    path = Path(os.getenv("RAG_DATABASE_PATH", str(Path(__file__).parent / "data" / "knowledge.sqlite3")))
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=10)
    connection.execute("CREATE TABLE IF NOT EXISTS knowledge (id INTEGER PRIMARY KEY CHECK(id=1), payload TEXT NOT NULL)")
    return connection


def _snapshot():
    with closing(_connect()) as connection:
        row = connection.execute("SELECT payload FROM knowledge WHERE id=1").fetchone()
    return json.loads(row[0]) if row else None


def _metadata(snapshot):
    if not snapshot:
        return {"ready": False}
    return {"ready": True, **{key: snapshot[key] for key in (
        "filename", "indexed_at", "model", "chunk_count", "sha256"
    )}}


def status():
    return _metadata(_snapshot())


# Mantém o título de cada tópico junto dos trechos, inclusive em seções longas.
def split_text(text: str) -> list[str]:
    sections = re.split(r"(?m)(?=^\d+[.)]\s+\S)", text.replace("\r\n", "\n"))
    chunks = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        heading = section.split("\n", 1)[0][:160] if re.match(r"^\d+[.)]\s", section) else ""
        start = 0
        while start < len(section):
            end = min(start + 1200, len(section))
            if end < len(section):
                boundary = max(section.rfind("\n", start + 600, end), section.rfind(" ", start + 600, end))
                if boundary > start:
                    end = boundary
            fragment = section[start:end].strip()
            chunks.append((heading + "\n" if start and heading else "") + fragment)
            if end == len(section):
                break
            start = end - 180
    return chunks


def _unit(vector):
    if not isinstance(vector, list) or not vector or any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
        for value in vector
    ):
        raise RagError("O provedor retornou embeddings inválidos.")
    norm = math.sqrt(sum(value * value for value in vector))
    if not norm or not math.isfinite(norm):
        raise RagError("O provedor retornou embeddings inválidos.")
    return [value / norm for value in vector]


def _tls_context():
    # Respeita um bundle explícito; por padrão, valida pela confiança nativa do SO.
    cafile = os.getenv("SSL_CERT_FILE")
    capath = os.getenv("SSL_CERT_DIR")
    if cafile or capath:
        return ssl.create_default_context(cafile=cafile, capath=capath)
    return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)


# Usa a mesma chave do chat; o modelo de embeddings é independente do modelo de conversa.
def embed(texts: list[str], model: str) -> list[list[float]]:
    key = os.getenv("OPENROUTER_API_KEY", "")
    if not key:
        raise RagError("Configure OPENROUTER_API_KEY no servidor Python.", 503)
    vectors = []
    try:
        with httpx.Client(timeout=45.0, verify=_tls_context()) as client:
            for start in range(0, len(texts), 32):
                batch = texts[start:start + 32]
                response = client.post(
                    "https://openrouter.ai/api/v1/embeddings",
                    headers={"Authorization": f"Bearer {key}"},
                    json={"model": model, "input": batch, "encoding_format": "float"},
                )
                response.raise_for_status()
                items = sorted(response.json()["data"], key=lambda item: item["index"])
                if [item['index'] for item in items] != list(range(len(batch))):
                    raise RagError("O provedor retornou uma quantidade inesperada de embeddings.")
                vectors.extend(_unit(item["embedding"]) for item in items)
    except httpx.HTTPStatusError as exc:
        messages = {
            401: "A OpenRouter recusou a chave de API. Confira OPENROUTER_API_KEY no servidor.",
            402: "A OpenRouter informou saldo insuficiente para gerar embeddings.",
            403: "A OpenRouter negou acesso ao modelo de embeddings. Confira as permissões da chave.",
            400: "A OpenRouter rejeitou a solicitação de embeddings. Confira OPENROUTER_EMBEDDING_MODEL; use um modelo de embeddings, não openrouter/auto.",
            404: "O modelo ou endpoint de embeddings não foi encontrado na OpenRouter.",
            429: "O limite de requisições da OpenRouter foi atingido. Tente novamente mais tarde.",
        }
        raise RagError(messages.get(exc.response.status_code,
            f"A OpenRouter falhou ao gerar embeddings (HTTP {exc.response.status_code}). Tente novamente mais tarde.")) from exc
    except httpx.TimeoutException as exc:
        raise RagError("A OpenRouter demorou para gerar embeddings. Tente novamente mais tarde.") from exc
    except httpx.ConnectError as exc:
        if "CERTIFICATE_VERIFY_FAILED" in str(exc):
            raise RagError("Falha no certificado HTTPS ao conectar à OpenRouter. Configure SSL_CERT_FILE com uma cadeia de certificados confiável no servidor Python. Trocar o modelo não resolve essa falha.") from exc
        raise RagError("Não foi possível conectar à OpenRouter. Confira a rede e o proxy do servidor Python.") from exc
    except httpx.HTTPError as exc:
        raise RagError("A conexão com a OpenRouter falhou durante a geração de embeddings.") from exc
    except (ValueError, KeyError, TypeError) as exc:
        raise RagError("O provedor retornou uma resposta de embeddings inválida.") from exc
    if len({len(vector) for vector in vectors}) != 1:
        raise RagError("O provedor retornou dimensões de embeddings inconsistentes.")
    return vectors


# Gera o índice completo antes de trocar a base, em uma única transação.
def ingest(filename: str, content: bytes):
    if not filename.lower().endswith('.txt'):
        raise RagError("Envie um arquivo .txt.", 400)
    if len(content) > MAX_BYTES:
        raise RagError("O arquivo deve ter no máximo 256 KB.", 413)
    try:
        text = content.decode("utf-8-sig").strip()
    except UnicodeDecodeError as exc:
        raise RagError("Salve o TXT com codificação UTF-8 antes de enviar.", 400) from exc
    if not text or any(ord(char) < 32 and char not in '\n\r\t' for char in text):
        raise RagError("O arquivo deve conter texto válido e não pode estar vazio.", 400)
    chunks = split_text(text)
    if len(chunks) > 256:
        raise RagError("O texto gerou muitos trechos. Reduza o conteúdo do arquivo.", 400)
    if not INGEST_LOCK.acquire(blocking=False):
        raise RagError("Já existe um arquivo em processamento. Aguarde e tente novamente.", 409)
    try:
        model = os.getenv("OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small")
        digest = hashlib.sha256(content).hexdigest()
        previous = _snapshot()
        if previous and previous['sha256'] == digest and previous['model'] == model:
            return {**_metadata(previous), "unchanged": True}
        vectors = embed(chunks, model)
        snapshot = {
            "filename": filename.replace('\\', '/').split('/')[-1][:200],
            "indexed_at": datetime.now(timezone.utc).isoformat(),
            "model": model, "sha256": digest, "chunk_count": len(chunks),
            "chunks": [{"text": text, "vector": vector} for text, vector in zip(chunks, vectors, strict=True)],
        }
        with closing(_connect()) as connection, connection:
            connection.execute("INSERT OR REPLACE INTO knowledge(id, payload) VALUES (1, ?)",
                               (json.dumps(snapshot, ensure_ascii=False),))
        return {**_metadata(snapshot), "unchanged": False}
    finally:
        INGEST_LOCK.release()


# A consulta usa o modelo salvo no índice, mesmo se a configuração tiver mudado.
def search(question: str):
    if not question.strip():
        raise RagError("Informe uma pergunta para pesquisar.", 400)
    snapshot = _snapshot()
    if snapshot is None:
        return {"ready": False, "trechos": [], "mensagem": "A base de conhecimento ainda não foi enviada."}
    query = embed([question], snapshot['model'])[0]
    results = []
    for index, chunk in enumerate(snapshot['chunks']):
        if len(query) != len(chunk['vector']):
            raise RagError("O modelo retornou outra dimensão. Reindexe a base de conhecimento.")
        score = sum(a * b for a, b in zip(query, chunk['vector'], strict=True))
        if score >= 0.25:
            results.append({"trecho_id": index + 1, "texto": chunk['text'], "similaridade": round(score, 4)})
    results.sort(key=lambda item: item['similaridade'], reverse=True)
    return {"ready": True, "fonte": snapshot['filename'], "indexado_em": snapshot['indexed_at'],
            "trechos": results[:5]}
