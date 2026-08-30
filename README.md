# Tech Fine-Tuning

Serviço responsável por preparar o dataset médico, executar QLoRA com Unsloth e publicar um
adaptador versionado para o `tech-ai`. Ele consome o contrato do `tech-ingestao`; não conhece XML
do MedQuAD, ChromaDB nem a aplicação que fará inferência.

O primeiro modelo escolhido é o `Qwen3-4B-Instruct-2507`, quantizado em 4 bits. Ele é muito menor
que o modelo de 17B testado anteriormente e o perfil inicial limita o treinamento a 50 passos.

## Fluxo

```text
tech-ingestao/artifacts/dataset
        │ prepare-sft
        ▼
artifacts/sft/{train,validation,test}.jsonl
        │ train + perfil TOML
        ▼
artifacts/training/<run>/{adapter,checkpoints,manifestos}
        │
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

## 2. Validar uma run sem GPU

O perfil pequeno está em [`configs/qwen3-4b/smoke.toml`](configs/qwen3-4b/smoke.toml). Para validar
configuração, dataset, hashes, procedência Git e destino sem baixar o modelo:

```powershell
uv run tech-fine-tuning train --dry-run
```

Por padrão, a execução local exige um commit identificável e um repositório limpo. Durante uma
experiência ainda não commitada, `--allow-dirty` libera somente essa proteção; a run continua
registrando o commit atual.

## 3. Treinar localmente

Instale o extra de GPU somente na máquina que fará treinamento:

```powershell
uv sync --dev --extra training --locked
uv run tech-fine-tuning diagnose --require-training
uv run tech-fine-tuning train
```

O extra fixa `unsloth==2026.8.22` no `uv.lock`. O diagnóstico mostra GPU, memória, driver, CUDA,
BF16 e versões efetivas dos pacotes antes de permitir o treinamento.

Para retomar uma interrupção:

```powershell
uv run tech-fine-tuning train `
  --resume-from-checkpoint artifacts/training/qwen3-4b-smoke/checkpoints/checkpoint-40
```

O perfil completo está em [`configs/qwen3-4b/full.toml`](configs/qwen3-4b/full.toml) e só deve ser
usado depois que o smoke test e a avaliação forem aprovados.

## 4. Treinar com Docker

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
artifacts/training/qwen3-4b-smoke/
├── adapter/                 Adaptador PEFT e tokenizer
├── checkpoints/             Pontos de retomada
├── run-manifest.json        Configuração, ambiente, métricas e status
└── model-manifest.json      Contrato de entrega ao tech-ai
```

O treinamento usa apenas `train` e `validation`. O split `test` permanece reservado para a
avaliação final. Pesos, datasets, cache e checkpoints são ignorados pelo Git.

## Configuração

As opções de treinamento ficam nos TOML versionados. Caminhos podem ser substituídos por:

- `TECH_FINE_TUNING_CANONICAL_DATASET_PATH`;
- `TECH_FINE_TUNING_SFT_OUTPUT_PATH`;
- `TECH_FINE_TUNING_TRAINING_CONFIG_PATH`;
- `TECH_FINE_TUNING_TRAINING_OUTPUT_PATH`;
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
