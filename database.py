"""Conexão com o banco: MySQL quando DATABASE_URL existe; senão, SQLite local.

Hospedagens gratuitas apagam o disco a cada reinício, então em produção a base
de conhecimento e as cotas ficam no MySQL. As consultas usam "?" como marcador
e são adaptadas ao driver.
"""

import os
import sqlite3
import ssl
from contextlib import closing, contextmanager
from pathlib import Path
from threading import Lock
from urllib.parse import unquote, urlsplit

import pymysql

Error = (sqlite3.Error, pymysql.Error)
_databases = set()
_created = set()
_create_lock = Lock()


def is_mysql() -> bool:
    return bool(os.getenv("DATABASE_URL"))


def _mysql():
    url = urlsplit(os.getenv("DATABASE_URL"))
    name = unquote(url.path.lstrip("/")) or "portfolio_ai"
    local = url.hostname in ("localhost", "127.0.0.1", "::1")
    connection = pymysql.connect(
        host=url.hostname, port=url.port or 3306,
        user=unquote(url.username or ""), password=unquote(url.password or ""),
        # Bancos em nuvem (ex.: TiDB Cloud) exigem TLS com certificado válido.
        ssl=None if local else ssl.create_default_context(),
        connect_timeout=10, read_timeout=30, write_timeout=30, charset="utf8mb4",
    )
    if name not in _databases:
        with connection.cursor() as cursor:
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{name.replace('`', '')}`")
        _databases.add(name)
    connection.select_db(name)
    return connection


class _Cursor:
    def __init__(self, cursor, mysql):
        self.cursor, self.mysql = cursor, mysql

    def execute(self, sql, params=()):
        self.cursor.execute(sql.replace("?", "%s") if self.mysql else sql, params)
        return self.cursor


@contextmanager
def transaction(sqlite_path: Path, schema: dict[str, list[str]], timeout=10):
    """Abre uma transação; `schema` tem as tabelas para "sqlite" e "mysql"."""
    mysql = is_mysql()
    if mysql:
        connection = _mysql()
    else:
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(sqlite_path, timeout=timeout)
    with closing(connection):
        cursor = _Cursor(connection.cursor(), mysql)
        # As tabelas são criadas uma vez por processo e banco.
        statements = tuple(schema["mysql" if mysql else "sqlite"])
        key = (os.getenv("DATABASE_URL") if mysql else str(sqlite_path), statements)
        if key not in _created:
            with _create_lock:
                for statement in statements:
                    cursor.execute(statement)
                connection.commit()
                _created.add(key)
        try:
            yield cursor
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
