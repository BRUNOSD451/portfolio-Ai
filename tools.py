"""Tools de consulta da IA pública do portfólio via backend Node.

Dados estruturados são consultados pela API Node, somente para leitura.
A busca RAG recupera trechos do TXT indexado no servidor Python.
"""

import json
from typing import Annotated

from langchain_core.tools import tool, ToolException
from pydantic import Field

from backend import get, normalize, owner
import database
import rag


# Converte os dados em texto JSON, preservando acentos para a resposta da tool.
def _json(data) -> str:
    return json.dumps(data, ensure_ascii=False)


# Seleciona apenas ID, nome e descrição para resumir um projeto na pesquisa.
def _project_summary(project: dict) -> dict:
    return {key: project.get(key) for key in ("id", "nome", "descricao")}


# Lista projetos pela API Node e filtra por nome, descrição ou tecnologia,
# ignorando caixa e acentos. Sem termo, retorna todos os resumos.
@tool(description=(
    "Pesquisa projetos por nome, descrição, objetivo ou tecnologia. Use para listar "
    "o portfólio ou encontrar trabalhos relevantes ao visitante. Deve retornar IDs, "
    "nomes e resumos para consultar detalhes depois. Termo vazio lista todos os projetos públicos."
))
def pesquisar_projetos(
    termo: Annotated[str, Field(description="Texto de busca; vazio para listar todos.")] = "",
) -> str:
    projects = get("/projects/catalog", list)
    term = normalize(termo)
    if not term:
        return _json([_project_summary(project) for project in projects])
    matches = []
    for project in projects:
        text = normalize(f"{project.get('nome') or ''} {project.get('descricao') or ''}")
        if term in text:
            matches.append(_project_summary(project))
            continue
        if any(term in normalize(tech['nome']) for tech in project['tecnologias']):
            matches.append(_project_summary(project))
    return _json(matches)


# Busca pelo ID no catálogo, reutilizado durante a requisição do chat.
@tool(description=(
    "Consulta nome, descrição e tecnologias de um projeto no banco. Use após "
    "identificar o projeto na pesquisa. Para relatos sobre participação do autor, "
    "desafios e decisões técnicas, use buscar_conhecimento_pessoal."
))
def obter_detalhes_projeto(
    projeto_id: Annotated[int, Field(gt=0, description="ID retornado pela pesquisa de projetos.")],
) -> str:
    details = next((p for p in get("/projects/catalog", list) if p['id'] == projeto_id), None)
    if details is None:
        raise ToolException("Registro não encontrado no backend.")
    return _json({
        "id": projeto_id,
        "nome": details['nome'],
        "descricao": details.get('descricao'),
        "tecnologias": [{"nome": tech['nome']} for tech in details['tecnologias']],
    })


# Filtra tecnologias pelo nome e cruza com o catálogo completo de projetos.
# Sem termo, retorna todas com seus projetos associados.
@tool(description=(
    "Consulta tecnologias cadastradas no banco e os projetos em que foram usadas. "
    "Use para perguntas sobre conhecimentos técnicos e exemplos práticos. Termo vazio "
    "lista todas. Não infira nível ou anos de experiência pela presença de uma tecnologia."
))
def consultar_tecnologias(
    termo: Annotated[str, Field(description="Tecnologia; vazio para listar todas.")] = "",
) -> str:
    term = normalize(termo)
    technologies = [
        tech for tech in get("/technologies/get", list)
        if term in normalize(tech['nome'])
    ]
    if not technologies:
        return _json([])
    usages = {}
    for project in get("/projects/catalog", list):
        for tech in project['tecnologias']:
            usages.setdefault(tech['nome'], []).append({"id": project['id'], "nome": project['nome']})
    return _json([
        {"id": tech['id'], "nome": tech['nome'],
         "projetos": usages.get(tech['nome'], [])}
        for tech in technologies
    ])


# Consulta o proprietário de ID 1 e retorna nome e apresentação em texto.
@tool(description=(
    "Consulta o perfil básico no banco: nome e descrição de apresentação. "
    "Use para uma apresentação geral do autor. Para aprofundar sua história, formação "
    "ou carreira, use buscar_conhecimento_pessoal; para contato, consultar_contatos."
))
def consultar_perfil() -> str:
    user = owner()
    return _json({key: user.get(key) for key in ("nome", "descricao")})


# Busca trechos próximos da pergunta na base pessoal indexada via RAG.
@tool(description=(
    "Busca via RAG nos textos e documentos pessoais publicados pelo autor. Use para "
    "formação acadêmica, experiências profissionais, cursos, certificações, idiomas, "
    "trajetória, interesses e preferências profissionais, além de relatos sobre "
    "participação, desafios e decisões em projetos. A pergunta deve incluir o assunto "
    "e, quando houver, o nome do projeto ou empresa. Deve retornar trechos relevantes "
    "com identificação da fonte e data de atualização quando disponível. Não invente "
    "informações quando não houver evidência. Para dados estruturados de projetos, "
    "tecnologias, perfil básico e contato, use as tools do banco."
))
def buscar_conhecimento_pessoal(
    pergunta: Annotated[str, Field(
        min_length=1, max_length=4000,
        description="Pergunta autossuficiente para busca nos documentos pessoais, com nomes e contexto relevantes.",
    )],
) -> str:
    try:
        return _json(rag.search(pergunta))
    except rag.RagError as exc:
        raise ToolException(str(exc)) from exc
    except (OSError, *database.Error) as exc:
        raise ToolException("Não foi possível consultar a base de conhecimento.") from exc


# Retorna somente e-mail e telefone do proprietário, tratados como contatos públicos.
# Campos não preenchidos retornam null; a função não envia mensagens.
@tool(description=(
    "Consulta e-mail e telefone cadastrados no banco e autorizados para exibição "
    "pública pelo autor. Use quando o visitante quiser entrar em contato. "
    "Apenas fornece os canais; não envia mensagens."
))
def consultar_contatos() -> str:
    user = owner()
    return _json({key: user.get(key) for key in ("email", "telefone")})


# Mídias e currículo não são expostos ao agente: o chat responde somente em texto.
PORTFOLIO_TOOLS = [
    pesquisar_projetos,
    obter_detalhes_projeto,
    consultar_tecnologias,
    consultar_perfil,
    buscar_conhecimento_pessoal,
    consultar_contatos,
]

# Falhas de consulta viram mensagens para a IA, sem expor respostas internas do backend.
for portfolio_tool in PORTFOLIO_TOOLS:
    portfolio_tool.handle_tool_error = lambda error: _json({"erro": str(error)})
