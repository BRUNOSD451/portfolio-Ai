# Portfolio AI API

API Python com FastAPI, LangChain e RAG. Expõe `POST /chat` e `GET/POST /knowledge`; a documentação automática do FastAPI está desativada.

Requer Python 3.11 ou superior. A API Node deve estar disponível para validar sessões administrativas; `PORTFOLIO_BACKEND_URL` é usado tanto nas consultas do agente quanto nessa validação.

## Configuração

1. Crie e ative um ambiente virtual dentro de `portfolioAi`:
   - Windows: `python -m venv .venv` e `.venv\\Scripts\\Activate.ps1`
2. Instale as dependências: `pip install -r requirements.txt`
3. Preencha `OPENROUTER_API_KEY` no arquivo `.env` com sua chave da OpenRouter.
4. (Opcional) Ajuste `OPENROUTER_MODEL` no `.env`.

## Executar

`uvicorn main:app --reload --host 0.0.0.0 --port 8000`

O servidor ficará disponível em `http://localhost:8000`. O frontend de gestão usa a API Node na porta 3000; mantenha sua configuração. Um cliente do chat deve chamar `POST /chat` na porta 8000.

## Tools do portfólio

As tools de dados estruturados fazem somente GET no backend Node, sem conexão direta com o banco. Mídias e link do currículo não são expostos ao agente, que responde somente em texto.
Inicie também o `portfolioBackEnd` e configure `PORTFOLIO_BACKEND_URL` no `.env` da IA
se ele não estiver em `http://localhost:3000`. Não é necessário alterar o `.env` para usar esse padrão.

Projetos usam `/projects/get` e `/projects/get/:id`; tecnologias usam `/technologies/get`.
A busca faz correspondência parcial, ignorando caixa e acentos. Como as rotas atuais não
oferecem filtros e vínculos em lote, buscas por tecnologia consultam detalhes dos projetos.
Perfil e contatos usam `/user/get`, selecionando apenas o proprietário de ID 1
e retornando somente os campos da respectiva tool. E-mail e telefone desse cadastro são
tratados como contatos públicos; o schema atual não possui controle de visibilidade.

As respostas são strings JSON: listas vazias indicam nenhum resultado, `null` indica campo
não preenchido e `{"erro": "..."}` indica falha na consulta. O timeout é de 10 segundos por
requisição. `buscar_conhecimento_pessoal` consulta a base RAG persistida no Python.

Execute `python -m unittest -v test_security` para verificar autenticação, limites e fluxo do agente com integrações simuladas. Para verificar a sintaxe, execute `python -m compileall -q main.py backend.py tools.py rag.py knowledge_routes.py security.py`.

`main.py` define o chat e seu limite de consultas; `tools.py` registra as seis ferramentas disponíveis; `backend.py` concentra as leituras da API Node; `rag.py` cuida da indexação e busca; `knowledge_routes.py` expõe o upload administrativo. `consultar_perfil` não recebe argumentos, pois sempre retorna o perfil básico do proprietário.

## Base de conhecimento (RAG)

1. Configure `OPENROUTER_API_KEY` e, opcionalmente, `OPENROUTER_EMBEDDING_MODEL`
   (padrão `openai/text-embedding-3-small`). O uso de embeddings consome créditos da OpenRouter.
2. Inicie a API Python com o comando acima. Na aba Conhecimento,
   selecione um TXT UTF-8 de até 256 KB e pressione **Enviar base de conhecimento**.
3. O app usa `EXPO_PUBLIC_AI_API_URL` para o Python; sem ela, usa o host do backend Node
   na porta 8000. `EXPO_PUBLIC_API_URL` continua apontando para o Node na porta 3000.
   No celular, use o IP acessível do computador e mantenha ambos na mesma rede.
4. Para acesso web por IP de rede, inclua a origem exata do frontend em `CORS_ORIGINS`
   no Python, separando múltiplas origens por vírgula. Reinicie o Python após alterar o `.env`.

O frontend envia multipart com campo `arquivo` e `Authorization: Bearer <token>` da sessão criada no backend Node.
`GET /knowledge` retorna nome, data e quantidade de trechos da base ativa.
As duas rotas de conhecimento exigem autenticação. A sessão é validada no Node antes de ler o upload. A tool do chat apenas consulta a base; `/chat` continua público com limites de frequência, concorrência, tamanho e cota diária persistida. [Detalhes e primeiro acesso](../portfolioBackEnd/SEGURANCA.md).

O TXT é dividido por tópicos numerados e por janelas de até 1.200 caracteres, com
sobreposição de 180 caracteres. Títulos são repetidos nos trechos de uma mesma seção.
Embeddings são gerados em lotes de 32 e armazenados, junto dos textos, modelo e fonte,
no MySQL de `DATABASE_URL` (tabelas `knowledge_meta` e `knowledge_chunks`, criadas
automaticamente, assim como a cota diária `budget`). Sem `DATABASE_URL`, ficam em
`data/knowledge.sqlite3` e `data/security.sqlite3`; `RAG_DATABASE_PATH` muda o local.
Em hospedagem com disco temporário (Render gratuito), use sempre `DATABASE_URL`.
`python migrate_local_data.py` copia uma base no formato antigo do SQLite para o destino atual.
O banco contém os textos pessoais e deve ser incluído em backups privados.
O SQLite e o arquivo local `base_conhecimento_bruno.txt` estão no `.gitignore`: preserve-os localmente, mas não são necessários para instalar ou iniciar o projeto. A base pode ser enviada novamente pelo app.

Cada upload substitui a base inteira, sem acumular versões. A troca é transacional:
falhas de validação ou embeddings preservam a base anterior. O mesmo conteúdo com o
mesmo modelo não gera embeddings novamente. O limite é de 256 trechos por arquivo.
Execute uma instância de processamento: a proteção contra uploads simultâneos é local
ao processo, não um bloqueio distribuído entre vários servidores.

Nas perguntas pessoais, a tool gera o embedding da pergunta com o modelo salvo no
índice e recupera até cinco trechos por similaridade de cosseno (mínimo 0,25).
Esse limiar é inicial e pode precisar de ajuste conforme perguntas reais. Os trechos
são evidências para o modelo responder, não garantia de que a resposta esteja neles.
Sem base ou sem trechos relevantes, a IA informa a falta de informação. A data de
indexação não substitui as datas dos fatos escritos no TXT. O conteúdo enviado deve
ser apropriado para as respostas públicas do portfólio.

A integração de embeddings segue o contrato da OpenRouter:
https://openrouter.ai/docs/api/api-reference/embeddings/submit-an-embedding-request

### HTTPS em rede corporativa

O cliente de embeddings usa `truststore` para validar HTTPS com os certificados
confiáveis do sistema operacional (no Windows, a validação nativa do Windows).
A verificação do certificado e do hostname permanece ativa. Se `SSL_CERT_FILE` ou
`SSL_CERT_DIR` estiver definido, esse bundle explícito tem prioridade.

Se a conexão com a OpenRouter ainda falhar com `CERTIFICATE_VERIFY_FAILED`, configure
`SSL_CERT_FILE` no `.env` apontando para um bundle PEM confiável fornecido pela
administração da rede e reinicie a API. O erro `Basic Constraints of CA cert not
marked critical` indica que o certificado da cadeia não atende à validação da CA;
é necessário um certificado/cadeia válido. A aplicação não desabilita a verificação
HTTPS. Enquanto a conexão falhar, o upload retorna erro e preserva a base anterior.

## Rota

### Consultas, desempenho e diagnóstico

O chat permite até três rodadas com ferramentas e seis execuções de ferramentas.
Ao atingir um desses limites, faz uma chamada final sem ferramentas para responder
com os resultados disponíveis. Todas as rodadas, incluindo a finalização, permanecem
dentro do timeout total de 75 segundos.

As ferramentas de projetos usam `GET /projects/catalog` do Node, que reúne nome,
descrição e tecnologias em três consultas ao banco, independentemente da quantidade
de projetos. As leituras bem-sucedidas do Node são reutilizadas apenas durante a
requisição atual do chat. Reinicie/atualize o backend Node junto com a API Python,
pois essa rota é necessária para as novas ferramentas.

Timeouts retornam HTTP 504; falhas do provedor ou do protocolo de ferramentas,
HTTP 502; falhas internas inesperadas, HTTP 500. Esses erros incluem `X-Request-ID`
para correlação com logs do servidor. Os logs registram tipo de erro e localização
no código, sem mensagens de exceção, conteúdo de conversas ou credenciais.

Execute `python -m unittest -v test_security test_chat_flow` para verificar o fluxo,
os limites, o tratamento de erros e a reutilização do catálogo sem chamadas reais de IA.

`POST /chat`

Requisição JSON:

`{"message":"Olá!"}`

O campo opcional `history` permite continuar a conversa:

```json
{"message":"Quais tecnologias ele usa?","history":[{"role":"user","content":"Fale do projeto X"},{"role":"assistant","content":"O projeto X é um portfólio."}]}
```

O frontend envia apenas trocas concluídas, com até seis pares de pergunta/resposta
e 12.000 caracteres no total. Cada mensagem do histórico aceita até 4.000 caracteres;
respostas maiores são recortadas no contexto. Falhas, cancelamentos e novas tentativas
não adicionam turnos à memória. O contexto fica na memória da página e é perdido ao
recarregá-la; não é persistido no servidor nem compartilhado entre visitantes.
O histórico é validado antes de reservar a cota diária e serve como contexto,
não como fonte verificada de fatos sobre Bruno. Clientes sem `history` continuam aceitos.

Resposta JSON:

`{"answer":"Olá! Como posso ajudar?"}`
