"""Copia a base de conhecimento do formato SQLite antigo para o formato atual.

O destino é o MySQL de DATABASE_URL ou, sem ele, o próprio SQLite local.
Uso: `python migrate_local_data.py`. Não gera embeddings nem consome créditos.
"""

import json
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

from dotenv import load_dotenv

import rag

load_dotenv()
source = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "data" / "knowledge.sqlite3")
with closing(sqlite3.connect(f"file:{source}?mode=ro", uri=True)) as connection:
    # Formato anterior: um único JSON com metadados e trechos.
    row = connection.execute("SELECT payload FROM knowledge WHERE id=1").fetchone()
if not row:
    sys.exit("Nenhuma base encontrada no SQLite.")
snapshot = json.loads(row[0])
rag.save(snapshot)
print(f"Base '{snapshot['filename']}' copiada: {snapshot['chunk_count']} trechos.")
