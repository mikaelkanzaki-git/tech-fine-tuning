# Instruções de arquitetura

Este serviço adota a Arquitetura em Camadas Pragmática descrita em
`docs/architecture/pragmatic-layered-architecture.md`.

Ao criar ou alterar código:

- coloque estruturas de dados e enums em `models/`;
- coloque preparação de SFT, treinamento e avaliação em `services/`;
- coloque leitura de datasets, Unsloth e outros SDKs em `integrations/`;
- coloque settings e montagem de dependências em `config/`;
- crie `repositories/` ou `api/` somente quando houver responsabilidade real;
- não crie diretórios genéricos `ports/`, `adapters/`, `utils/` ou `helpers/`;
- mantenha o fluxo de dependência `runner/api -> services -> models`;
- não importe Unsloth, Transformers ou SDKs externos em `models/`;
- mantenha teste e avaliação isolados do split de treino;
- publique modelos conforme `schemas/model-artifact-manifest.schema.json`;
- não faça commit, push ou PR sem autorização explícita do usuário.

Antes de entregar mudanças, execute:

```powershell
uv run ruff check .
uv run mypy
uv run pytest
```
