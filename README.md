# Tech Fine-Tuning

Serviço responsável por preparar o dataset médico, executar QLoRA com Unsloth e publicar um
adaptador versionado para o `tech-ai`. Ele consome o contrato do `tech-ingestao`; não conhece XML
do MedQuAD, ChromaDB nem a aplicação que fará inferência.

O primeiro modelo escolhido é o `Qwen3-4B-Instruct-2507`, quantizado em 4 bits. Ele é muito menor
que o modelo de 17B testado anteriormente e o perfil inicial limita o treinamento a 50 passos.

## Arquitetura

O serviço usa a mesma Arquitetura em Camadas Pragmática dos demais projetos:

```text
src/tech_fine_tuning/
├── models/          DTOs, configurações e resultados internos
├── services/        Preparação, auditoria, treinamento e avaliação
├── integrations/    JSONL, sistema, manifestos e Unsloth
├── config/          Settings e composição
├── errors.py
└── runner.py
```

Para quem vem de Java/Spring: `models` representa DTOs e tipos do domínio, `services` contém os
casos de uso, `integrations` encapsula SDKs externos, `config` equivale à configuração da
aplicação e `runner.py` é a entrada executável. `repositories` e `api` não existem porque este
serviço não possui banco nem transporte HTTP.

O detalhamento está em
[`docs/architecture/pragmatic-layered-architecture.md`](docs/architecture/pragmatic-layered-architecture.md).

## Fluxo

```text
tech-ingestao/artifacts/dataset
        │ prepare-sft + validação de isolamento
        ▼
artifacts/sft-v2/{train,validation,test}.jsonl
        │ audit-sft + nova instrução segura e concisa
        ├──────────────► reports/data-quality/<auditoria>
        │ curate-sft
        ▼
artifacts/sft-candidate-v3/{train,validation,test}.jsonl
        │ audit-sft + amostragem balanceada + train
        ▼
artifacts/training/<run>/{adapter,checkpoints,manifestos}
        │ evaluate
        ▼
artifacts/evaluation/<run>/<split>/{respostas,métricas,revisão}
        │ modelo aprovado
        ▼
tech-ai
```

O código e os perfis são os mesmos em execução local, Docker, Colab, Azure ML ou Vertex AI. Os
provedores apenas entregam GPU, filesystem e variáveis de execução.

## Requisitos básicos

- Python 3.12;
- [uv](https://docs.astral.sh/uv/);
- `tech-ingestao` e `tech-fine-tuning` lado a lado;
- dataset canônico `1.1` já gerado.

Para preparar dados, testar e validar configurações sem instalar CUDA:

```powershell
uv sync --dev --locked
```

## 1. Gerar o dataset SFT

```powershell
uv run tech-fine-tuning prepare-sft
```

O comando cria `train.jsonl`, `validation.jsonl`, `test.jsonl` e `manifest.json` em
`artifacts/sft`. O serviço valida novamente curadoria, PII, contagens, hashes e ausência de
vazamento entre documentos. O contrato está em
[`docs/data/sft-dataset.md`](docs/data/sft-dataset.md).

Antes de treinar, audite o material conversacional sem modificá-lo:

```powershell
uv run tech-fine-tuning audit-sft `
  --source "artifacts/sft" `
  --output "reports/data-quality/sft-baseline" `
  --max-sequence-length 2048
```

O comando verifica os hashes do SFT e registra boilerplate recorrente, respostas que apenas apontam
para outros recursos, repetição de frases, provável estouro de contexto, perguntas duplicadas e
perguntas presentes em mais de um split. Ele gera `audit-summary.json`, `issues.jsonl`, `report.md`
e `audit-manifest.json`. A estimativa de contexto usa quatro caracteres por token somente como
triagem; o tokenizer do modelo continua sendo a fonte exata. Consulte
[`docs/data/sft-audit.md`](docs/data/sft-audit.md).

Depois de corrigir o isolamento na origem, aplique a curadoria conservadora em uma nova derivação:

```powershell
uv run tech-fine-tuning curate-sft `
  --source "artifacts/sft-v2" `
  --output "artifacts/sft-candidate-v3" `
  --system-prompt "You are a medical information assistant. Answer only the question asked with concise educational information grounded in reliable medical sources. If reliable information is unavailable, say that it is unknown instead of inventing details. Do not prescribe medication, diagnose a patient, or replace a qualified healthcare professional."
```

Ela remove somente texto não instrutivo por regras determinísticas, registra todos os excluídos e
não sintetiza fatos médicos. Consulte
[`docs/data/sft-curation.md`](docs/data/sft-curation.md).

O candidato v3 parte do `sft-v2`, o snapshot que já eliminou perguntas repetidas entre splits. A
opção `--system-prompt` troca apenas a mensagem de sistema, sem reescrever as respostas médicas, e
registra o novo texto e seu SHA-256 no manifesto derivado.

## 2. Validar uma run sem GPU

O perfil pequeno está em [`configs/qwen3-4b/smoke.toml`](configs/qwen3-4b/smoke.toml). Para validar
configuração, dataset, hashes, procedência Git e destino sem baixar o modelo:

```powershell
uv run tech-fine-tuning train --dry-run
```

Por padrão, a execução local exige um commit identificável e um repositório limpo. Durante uma
experiência ainda não commitada, `--allow-dirty` libera somente essa proteção; a run continua
registrando o commit atual.

O candidato v3 mantém o Qwen3-4B e os hiperparâmetros do smoke test, mas seleciona os 1.000
exemplos de maneira determinística e equilibrada entre as coleções do MedQuAD:

```powershell
uv run tech-fine-tuning train `
  --config "configs/qwen3-4b/candidate-v3-smoke.toml" `
  --source "artifacts/sft-candidate-v3" `
  --output "artifacts/training/qwen3-4b-candidate-v3" `
  --dry-run
```

O perfil registra `sampling_strategy = "balanced_by_source"`. A run também grava, para treino e
validação, quantos exemplos estavam disponíveis e quantos foram efetivamente selecionados por
coleção. Perfis antigos sem essa opção continuam usando `head` para preservar reprodutibilidade.

Remova `--dry-run` somente depois que a auditoria do candidato e a procedência Git estiverem
aprovadas.

## 3. Treinar localmente

Instale o extra de GPU somente na máquina que fará treinamento:

```powershell
uv sync --dev --extra training --locked
uv run tech-fine-tuning diagnose --require-training
uv run tech-fine-tuning train `
  --config "configs/qwen3-4b/candidate-v3-smoke.toml" `
  --source "artifacts/sft-candidate-v3" `
  --output "artifacts/training/qwen3-4b-candidate-v3"
```

O extra fixa `unsloth==2026.8.22` no `uv.lock`. O diagnóstico mostra GPU, memória, driver, CUDA,
BF16 e versões efetivas dos pacotes antes de permitir o treinamento. Em Windows e Linux, `torch`
e `torchvision` são obtidos do índice oficial `pytorch-cu130`; isso evita a instalação silenciosa
do wheel `+cpu` disponível no PyPI comum.

Depois de atualizar um checkout que já possuía a variante CPU, force a sincronização uma vez:

```powershell
uv sync --dev --extra training --locked --reinstall-package torch --reinstall-package torchvision
uv run tech-fine-tuning diagnose --require-training
```

Para retomar uma interrupção:

```powershell
uv run tech-fine-tuning train `
  --config "configs/qwen3-4b/candidate-v3-smoke.toml" `
  --source "artifacts/sft-candidate-v3" `
  --output "artifacts/training/qwen3-4b-candidate-v3" `
  --resume-from-checkpoint artifacts/training/qwen3-4b-candidate-v3/checkpoints/checkpoint-40
```

O perfil completo está em [`configs/qwen3-4b/full.toml`](configs/qwen3-4b/full.toml) e só deve ser
usado depois que o smoke test e a avaliação forem aprovados.

## 4. Comparar o modelo base com o fine-tuned

Depois que `train` terminar, faça primeiro uma avaliação pequena no split `validation`:

```powershell
uv run tech-fine-tuning evaluate `
  --manifest "artifacts/training/qwen3-4b-candidate-v3/model-manifest.json" `
  --source "artifacts/sft-candidate-v3" `
  --output "artifacts/evaluation/qwen3-4b-candidate-v3/sampling-validation" `
  --split validation `
  --sample-size 20 `
  --compare-base
```

No PowerShell, o acento grave `` ` `` deve ser o último caractere da linha. Não coloque `\`
antes dele nem antes das opções. O comando verifica o adaptador e a procedência do dataset, escolhe
uma amostra determinística e gera:

```text
artifacts/evaluation/qwen3-4b-candidate-v3/sampling-validation/
├── responses.jsonl          Respostas de referência, base e fine-tuned
├── automatic-metrics.json   Latência, respostas vazias e token F1 auxiliar
├── human-review.csv         Planilha para avaliação médica lado a lado
└── evaluation-manifest.json Entradas, hashes, parâmetros e resumo da execução
```

A avaliação usa por padrão a amostragem recomendada para o Qwen3 Instruct (`temperature=0.7`,
`top_p=0.8` e `top_k=20`) e `repetition_penalty=1.1`. A semente de cada pergunta é reaplicada no
modelo base e no fine-tuned, tornando a comparação reproduzível. Use `--greedy` apenas para
diagnóstico; todos os parâmetros efetivos ficam registrados no manifesto.

O token F1 apenas sinaliza diferenças; ele não comprova correção clínica. Preencha a planilha de
revisão sem olhar antecipadamente qual resposta venceu a métrica. Use `test` uma única vez, depois
que configuração e critérios estiverem estabilizados, para a decisão final. O procedimento e os
critérios estão em [`docs/training/evaluation.md`](docs/training/evaluation.md).

O diretório de saída deve estar vazio. Para repetir uma experiência, informe outro `--output`, por
exemplo `artifacts/evaluation/qwen3-4b-candidate-v3/sampling-validation-v2`.

Depois de preencher todas as notas, consolide a revisão em um diretório versionável separado dos
artefatos reproduzíveis:

```powershell
uv run tech-fine-tuning summarize-review `
  --review "artifacts/evaluation/qwen3-4b-candidate-v3/sampling-validation/human-review.csv" `
  --evaluation-manifest "artifacts/evaluation/qwen3-4b-candidate-v3/sampling-validation/evaluation-manifest.json" `
  --output "reports/evaluation/qwen3-4b-candidate-v3-validation"
```

O comando valida notas, preferências e `record_id`, preserva a planilha preenchida e gera:

```text
reports/evaluation/qwen3-4b-candidate-v3-validation/
├── human-review-completed.csv  Revisão humana preservada
├── human-review-summary.json   Métricas e resultado do quality gate
├── decision.md                 Decisão legível e próximo passo
└── review-manifest.json        Procedência e hashes dos arquivos
```

Uma decisão `rejected` impede o uso do split `test` e a publicação do adaptador. Nesse caso,
corrija dados ou treinamento e repita a avaliação em `validation` com um novo diretório de saída.

## 5. Treinar com Docker

O contêiner usa a imagem CUDA oficial do Unsloth e recebe a revisão Git no build:

```powershell
$revision = git rev-parse HEAD
docker build --build-arg VCS_REF=$revision -f docker/Dockerfile -t tech-fine-tuning:$revision .
docker volume create tech-fine-tuning-hf-cache
docker run --rm --gpus all --shm-size 8g `
  -v "${PWD}/artifacts:/workspace/tech-fine-tuning/artifacts" `
  -v "tech-fine-tuning-hf-cache:/workspace/cache/huggingface" `
  tech-fine-tuning:$revision python -m tech_fine_tuning diagnose --require-training
docker run --rm --gpus all --shm-size 8g `
  -v "${PWD}/artifacts:/workspace/tech-fine-tuning/artifacts" `
  -v "tech-fine-tuning-hf-cache:/workspace/cache/huggingface" `
  tech-fine-tuning:$revision python -m tech_fine_tuning train
```

Para congelar também a imagem-base, passe um digest imutável em
`--build-arg UNSLOTH_IMAGE=unsloth/unsloth@sha256:...`.

## Colab, Azure ML e Vertex AI

- Colab: abra [`notebooks/colab_train.ipynb`](notebooks/colab_train.ipynb), selecione uma GPU,
  substitua o commit e mantenha dataset/checkpoints no Google Drive.
- Azure ML: os modelos de ambiente e job estão em [`cloud/azure`](cloud/azure).
- Vertex AI: o Custom Job de exemplo está em [`cloud/vertex`](cloud/vertex).

Os passos de publicação da imagem, armazenamento e retomada estão em
[`docs/training/portable-execution.md`](docs/training/portable-execution.md).

## Artefatos de saída

```text
artifacts/training/qwen3-4b-candidate-v3/
├── adapter/                 Adaptador PEFT e tokenizer
├── checkpoints/             Pontos de retomada
├── run-manifest.json        Configuração, ambiente, métricas e status
└── model-manifest.json      Contrato de entrega ao tech-ai
```

O treinamento usa apenas `train` e `validation`. O split `test` permanece reservado para a
avaliação final. Pesos, datasets, cache, checkpoints e saídas de avaliação são ignorados pelo Git.

## Configuração

As opções de treinamento ficam nos TOML versionados. Caminhos podem ser substituídos por:

- `TECH_FINE_TUNING_CANONICAL_DATASET_PATH`;
- `TECH_FINE_TUNING_SFT_OUTPUT_PATH`;
- `TECH_FINE_TUNING_TRAINING_CONFIG_PATH`;
- `TECH_FINE_TUNING_TRAINING_OUTPUT_PATH`;
- `TECH_FINE_TUNING_EVALUATION_OUTPUT_PATH`;
- `TECH_FINE_TUNING_REVISION` em imagens ou jobs sem checkout Git.

Nunca grave tokens do Hugging Face, Azure ou Google nos perfis ou manifestos. Entregue-os pelo
gerenciador de segredos do ambiente.

## Validar o projeto

```powershell
uv run ruff check .
uv run mypy
uv run pytest
```

A arquitetura está descrita em
[`docs/architecture/pragmatic-layered-architecture.md`](docs/architecture/pragmatic-layered-architecture.md).
