"""Upload administrativo da base; o chat público só pode consultá-la."""

from fastapi import APIRouter, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

import database
import rag


router = APIRouter(prefix="/knowledge")


@router.get("")
def knowledge_status():
    try:
        return rag.status()
    except (OSError, *database.Error):
        raise HTTPException(503, "Não foi possível acessar a base de conhecimento.")


@router.post("")
async def upload_knowledge(arquivo: UploadFile):
    try:
        # Lê um byte além do limite para rejeitar arquivos grandes sem carregá-los inteiros.
        content = await arquivo.read(rag.MAX_BYTES + 1)
        return await run_in_threadpool(rag.ingest, arquivo.filename or "", content)
    except rag.RagError as exc:
        raise HTTPException(exc.status, str(exc)) from exc
    except (OSError, *database.Error) as exc:
        raise HTTPException(503, "Não foi possível salvar a base de conhecimento.") from exc
    finally:
        await arquivo.close()
