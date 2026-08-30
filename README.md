# Tech Fine-Tuning

Serviço responsável por transformar o dataset médico curado em exemplos de treinamento,
executar o fine-tuning com Unsloth e avaliar o modelo resultante. Ele consome o contrato
canônico do `tech-ingestao`; não conhece XML do MedQuAD, ChromaDB ou regras de ingestão.

## Estado atual

A primeira fatia executável gera um dataset conversacional de Supervised Fine-Tuning (SFT).
Treinamento, escolha do modelo base e integração com Unsloth serão adicionados após a validação
de GPU/CUDA ou a escolha de uma GPU em nuvem.

## Arquitetura

O projeto segue a Arquitetura em Camadas Pragmática usada nos demais serviços:

```text
tech-fine-tuning/
├── docs/
├── src/tech_fine_tuning/
│   ├── config/         Settings da aplicação
│   ├── integrations/   Leitura e escrita de datasets; futuramente Unsloth
│   ├── models/         Contratos internos de dados
│   ├── services/       Preparação de SFT; futuramente treino e avaliação
│   ├── errors.py
│   └── runner.py
└── tests/unit/
```

Pastas de API e repositories não existem porque esta fatia ainda não possui HTTP nem banco.
A decisão arquitetural está em
[`docs/architecture/pragmatic-layered-architecture.md`](docs/architecture/pragmatic-layered-architecture.md).

## Requisitos

- Python 3.12;
- [uv](https://docs.astral.sh/uv/);
- `tech-ingestao` e `tech-fine-tuning` lado a lado;
- dataset canônico `1.1` já gerado e com PII resolvida.

```text
tech-3/
├── tech-ingestao/artifacts/dataset/
└── tech-fine-tuning/
```

## Preparação

No diretório `tech-fine-tuning`:

```powershell
uv sync --dev
```

## Gerar o dataset SFT

Com os caminhos padrão:

```powershell
uv run tech-fine-tuning prepare-sft
```

Ou explicitamente:

```powershell
uv run tech-fine-tuning prepare-sft `
  --source "..\tech-ingestao\artifacts\dataset" `
  --output "artifacts\sft"
```

O comando cria `train.jsonl`, `validation.jsonl`, `test.jsonl` e `manifest.json`. Cada exemplo
possui mensagens `system`, `user` e `assistant`, além de metadados de rastreabilidade que não
fazem parte do texto de treinamento.

O serviço valida novamente:

- versão `1.1` do contrato canônico;
- status de curadoria aceito;
- PII como `not_detected` ou `redacted`;
- contagens declaradas no manifesto de origem;
- ausência de vazamento de documento, registro e conteúdo entre splits.

O contrato completo está em [`docs/data/sft-dataset.md`](docs/data/sft-dataset.md).

## Configuração

Os padrões podem ser substituídos por ambiente:

- `TECH_FINE_TUNING_CANONICAL_DATASET_PATH`;
- `TECH_FINE_TUNING_SFT_OUTPUT_PATH`;
- `TECH_FINE_TUNING_SYSTEM_PROMPT`.

A instrução de sistema também pode ser passada com `--system-prompt`. Ela fica registrada no
manifesto junto de seu SHA-256 para reproduzir exatamente a derivação.

## Publicação do modelo

Depois do treinamento e da avaliação, este serviço publicará os pesos ou adaptadores junto de
um `model-manifest.json`. Esse será o único contrato necessário para o `tech-ai` carregar o
modelo pronto. O JSON Schema planejado está em
[`schemas/model-artifact-manifest.schema.json`](schemas/model-artifact-manifest.schema.json).

## Validar o projeto

```powershell
uv run ruff check .
uv run mypy
uv run pytest
```

## Próximas etapas

1. confirmar GPU, CUDA, PyTorch e versão do Unsloth;
2. escolher um modelo base pequeno e compatível;
3. aplicar o chat template do tokenizer somente no carregamento para treino;
4. executar um smoke test de fine-tuning;
5. avaliar o checkpoint nos splits de validação e teste;
6. publicar o artefato e seu manifesto para consumo pelo `tech-ai`.
