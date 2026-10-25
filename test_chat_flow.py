import asyncio
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from fastapi import HTTPException
from openai import APIConnectionError

import backend
import main
import tools


class ChatFlowTests(unittest.TestCase):
    def test_final_answer_after_three_tool_rounds_and_six_calls(self):
        for calls_per_round in (1, 2, 8):
            with self.subTest(calls_per_round=calls_per_round):
                executed = []
                class Model:
                    def __init__(self, **kwargs): self.rounds = 0
                    def bind_tools(self, available):
                        async def invoke(messages):
                            self.rounds += 1
                            return SimpleNamespace(content="", tool_calls=[
                                {"name": "consultar_perfil", "args": {}, "id": f"{self.rounds}-{i}"}
                                for i in range(calls_per_round)
                            ])
                        return SimpleNamespace(ainvoke=invoke)
                    async def ainvoke(self, messages):
                        results = [message for message in messages if getattr(message, "type", None) == "tool"]
                        assert len(results) == min(3, (6 + calls_per_round - 1) // calls_per_round) * calls_per_round
                        return SimpleNamespace(content="Resposta final", tool_calls=[])
                def owner():
                    executed.append(True)
                    return {"nome": "Bruno"}
                with patch.dict(os.environ, OPENROUTER_API_KEY="test"), patch.object(main, "ChatOpenAI", Model), patch.object(tools, "owner", owner):
                    response = asyncio.run(main.chat(main.ChatRequest(message="Projetos?")))
                self.assertEqual(response.answer, "Resposta final")
                self.assertEqual(len(executed), min(3 * calls_per_round, 6))

    def test_errors_have_distinct_status_and_safe_logs(self):
        errors = [(TimeoutError("secret"), 504),
                  (APIConnectionError(message="secret", request=httpx.Request("POST", "https://example.test")), 502),
                  (main.AgentProtocolError("secret"), 502),
                  (RuntimeError("secret"), 500)]
        for error, status in errors:
            with self.subTest(status=status):
                class Model:
                    def __init__(self, **kwargs): pass
                    def bind_tools(self, available): return self
                    async def ainvoke(self, messages): raise error
                with patch.dict(os.environ, OPENROUTER_API_KEY="test"), patch.object(main, "ChatOpenAI", Model), self.assertLogs(main.logger) as logs:
                    with self.assertRaises(HTTPException) as caught:
                        asyncio.run(main.chat(main.ChatRequest(message="secret")))
                self.assertEqual(caught.exception.status_code, status)
                self.assertIn(caught.exception.headers["X-Request-ID"], logs.output[0])
                self.assertNotIn("secret", " ".join(logs.output))

    def test_catalog_reused_across_tools_but_not_requests(self):
        catalog = [{"id": 1, "nome": "Site", "descricao": "Portfolio", "tecnologias": [{"id": 2, "nome": "React"}]}]
        paths = []
        def get(url, **kwargs):
            path = httpx.URL(url).path
            paths.append(path)
            return httpx.Response(200, request=httpx.Request("GET", url),
                                  json=catalog if path.endswith("catalog") else [{"id": 2, "nome": "React"}])
        async def run():
            async with backend.request_cache():
                found = json.loads(await tools.pesquisar_projetos.ainvoke({"termo": "React"}))
                self.assertEqual(found[0]["id"], 1)
                details = json.loads(await tools.obter_detalhes_projeto.ainvoke({"projeto_id": 1}))
                self.assertEqual(details["descricao"], "Portfolio")
                techs = json.loads(await tools.consultar_tecnologias.ainvoke({}))
                self.assertEqual(techs[0]["projetos"], [{"id": 1, "nome": "Site"}])
            async with backend.request_cache():
                await tools.pesquisar_projetos.ainvoke({})
        with patch.object(backend.httpx, "get", get):
            asyncio.run(run())
        self.assertEqual(paths, ["/projects/catalog", "/technologies/get", "/projects/catalog"])
