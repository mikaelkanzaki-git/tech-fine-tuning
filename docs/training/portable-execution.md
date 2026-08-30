# Execução portátil do fine-tuning

## Princípio

A unidade reproduzível é formada por cinco itens:

1. commit do `tech-fine-tuning`;
2. perfil TOML versionado;
3. SHA-256 do manifesto SFT e de seus splits;
4. revisões imutáveis do modelo e tokenizer;
5. versões do runtime registradas no `run-manifest.json`.

Colab, Azure ML e Vertex AI não possuem lógica de treinamento própria. Todos chamam
`python -m tech_fine_tuning train`; assim, mudar de provedor não muda o experimento.

## Armazenamento persistente

Separe dois volumes:

- cache do Hugging Face, que pode ser descartado e recriado;
- saída da run, que contém checkpoints, adaptador e manifestos e deve ser persistente.

Nunca coloque `artifacts/sft`, checkpoints ou pesos dentro da imagem Docker. A imagem representa
o código; volumes ou object storage representam entradas e saídas.

## Google Colab

1. Copie `artifacts/sft` para o Google Drive.
2. Abra `notebooks/colab_train.ipynb` no Colab e selecione runtime com GPU.
3. Troque `PROJECT_REVISION` por um commit completo já enviado ao GitHub.
4. Ajuste `SFT_DATASET` e `TRAINING_OUTPUT` para diretórios do Drive.
5. Execute `diagnose` e somente depois a célula `train`.

Como o Colab pode encerrar a sessão, a saída deve permanecer no Drive. Para continuar, acrescente
`--resume-from-checkpoint <diretório-do-checkpoint>` à última célula. Não use `/content` para
checkpoints que precisem sobreviver à sessão.

## Azure Machine Learning

O template usa um environment construído pelo mesmo `docker/Dockerfile` e um `command job` com
input/output do tipo `uri_folder`.

```bash
az ml environment create --file cloud/azure/environment.yml
az ml job create --file cloud/azure/job.example.yml
```

Antes do envio:

- registre `artifacts/sft` como data asset `medquad-sft` ou altere o `path` do input;
- substitua `producer_commit` por um commit completo;
- substitua `compute` por um cluster com GPU compatível;
- use uma identidade gerenciada ou secret store para credenciais privadas.

O output montado pelo Azure recebe checkpoints e manifestos durante a execução, permitindo
recuperação mesmo se a instância for desalocada.

## Google Vertex AI

Construa e publique a imagem no Artifact Registry:

```bash
REVISION=$(git rev-parse HEAD)
IMAGE=REGION-docker.pkg.dev/PROJECT/REPOSITORY/tech-fine-tuning:$REVISION
docker build --build-arg VCS_REF=$REVISION -f docker/Dockerfile -t $IMAGE .
docker push $IMAGE
gcloud ai custom-jobs create --region=REGION --config=cloud/vertex/custom-job.example.yaml
```

No YAML, substitua imagem, bucket e commit. O Cloud Storage FUSE expõe buckets no caminho
`/gcs/BUCKET`; por isso o mesmo comando recebe diretórios comuns. A service account do job deve
ter apenas leitura no dataset e escrita no prefixo da run.

## Troca de infraestrutura

Para comparar duas execuções, confirme primeiro os hashes e revisões nos manifestos. Diferenças de
GPU, CUDA, PyTorch ou Unsloth aparecem no bloco `environment`. Não declare duas runs equivalentes
se perfil, dataset, revisão do modelo ou código produtor forem diferentes.

## Segurança e custos

- prefira o perfil `smoke` antes de reservar uma GPU maior ou executar o perfil completo;
- limite a retenção de checkpoints com `save_total_limit`;
- encerre instâncias e clusters ociosos;
- não registre tokens em notebooks, YAML, logs ou manifestos;
- confira a licença e as restrições dos dados antes de enviar datasets a outro provedor.
