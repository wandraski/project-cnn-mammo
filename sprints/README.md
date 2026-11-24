# Rodar Sprints 07–11

Comece por `sprints/sprint-07/<plataforma>/`. São notebooks de pesquisa, não para uso clínico. Não execute `notebooks/09_exportacao.ipynb` para publicar pesos: aquele caderno legado usa artefatos ilustrativos.

**O que cada caderno faz**

- **S07:** treina ResNet50 de referência (20 épocas, batch 32, 224 px, LR 1e-4 na execução experimental). Não treina a U-Net nem executa os 10 encoders dos slides.
- **S08:** treina a cabeça da ResNet50 ImageNet com backbone congelado (5 épocas, batch 16, 224 px, AdamW); não substitui fine-tuning RadImageNet ou calibração por fonte.
- **S09:** roda **um braço por vez**, por padrão `batch-16-fixed`. Use `batch-{8,16,32,64,128}-{fixed,linear}` para escolher outro; a configuração experimental é de 15 épocas. Smoke não testa se o batch nominal cabe. Comece pequeno e registre falta de VRAM em vez de mudar a configuração silenciosamente. Não roda ainda o resgate OC-SVM.
- **S10:** executa o gate sobre predições reais de validação fornecidas e grava contagem de rotas e cobertura automática. A geração integrada de escores da U-Net e do OC-SVM e a validação externa ainda faltam.
- **S11 (parcial):** ajusta temperatura e agrega TTA sobre quatro logits reais por amostra de validação. Não gera os logits nem reporta a validação do próprio ajuste como resultado externo.

## 1. Prepare os dados uma vez

Só use dados de desenvolvimento autorizados para a plataforma. Não suba o teste final, dados identificáveis, ou dados cuja licença proíba esse uso. Não há download implícito. A estrutura interna do subset do Kaggle não foi confirmada; se a versão anexada não tiver um manifesto compatível, esta etapa é obrigatória.

Pasta `desenvolvimento/`: imagens (PNG/JPEG ou DICOM com extensão `.dcm`/`.dicom`) e `manifest.csv` na raiz. Use este **cabeçalho**, uma linha por imagem:

```text
sample_key,patient_key,path,source,label,split
```

- `sample_key`: chave única da imagem; `patient_key`: chave pseudonimizada do paciente, igual para todas as incidências do mesmo paciente. Use os metadados originais; id da imagem ou nome da pasta não serve como id de paciente. Resolva duplicatas entre bases antes de rodar.
- `path`: caminho relativo à pasta `desenvolvimento/`; `source`: base de origem; `label`: 0 benigno, 1 maligno; `split`: `train` ou `val`. Ambos os splits precisam das duas classes. Preserve partições oficiais e mantenha qualquer teste reservado fora da pasta.
- O preflight recusa paths ausentes, pacientes em ambos os splits, duplicatas e `test`. Não reescreve os dados. Se seu CSV original usa outros nomes, converta explicitamente as colunas depois de conferir a origem de cada uma.

S10/S11 não precisam dos pixels, mas exigem arquivos separados nesta mesma pasta. **Não use valores de slides ou exemplos para preenchê-los.**

```text
sprint10_val.csv → sample_key,split,p_cnn,s_anom,unet_conf
sprint11_val.csv → sample_key,split,label,logit_original,logit_hflip,logit_plus5,logit_minus5
```

`split=val`; probabilidades/confianças da S10 em [0,1]; logits da S11 antes da sigmoide, com rótulos 0/1. O JSON da CNN sozinho não contém os três sinais da S10 nem os quatro logits da S11.

## 2. Escolha uma plataforma

Gere o pacote de código na raiz do checkout: `python scripts/package_sprints.py`. Sai em `outputs/mammo-codigo.zip`, só com Python e TOML. Não depende do repositório estar público.

**Colab:** copie o ZIP para `MyDrive/mammo/mammo-codigo.zip` e os dados para `MyDrive/mammo/desenvolvimento/`. Abra o `.ipynb` de `colab/`, escolha GPU para S07–S09, rode as células na ordem e autorize o Drive. A saída fica em `MyDrive/mammo/resultados/`. Não precisa de chave Kaggle se os dados já estiverem no Drive.

**Kaggle:** importe o `.ipynb` de `kaggle/`, habilite GPU para S07–S09 e use **Add Input** para anexar o subset e um input de código que contenha `mammo-codigo.zip`. Se a versão do subset não trouxer o manifesto, anexe `manifest.csv` junto ao input de código: os paths continuam relativos a `DATA`, sem precisar reenviar imagens. Se o layout mudar, ajuste `DATA` na primeira célula. Para S10/S11, aponte `DATA` para o input que contém o CSV de predições. Inputs são lidos no lugar, sem cópia; não precisa de API key. Habilite internet se o runtime precisar instalar decodificadores DICOM ou baixar pesos ImageNet. Em ambientes sem internet use dependências e pesos já disponíveis, ou rode o smoke com dados PNG/JPEG. A saída fica em `/kaggle/working/resultados/`; preserve a saída ao salvar a versão ou baixe-a.

**Modal:** execute na raiz do checkout; o notebook de `modal/` é um lançador local, não precisa de Jupyter para usar os comandos abaixo. Confira os custos/cota do workspace. Crie volumes uma vez e selecione um diretório contendo **somente** os dados de desenvolvimento autorizados.

```bash
python -m pip install modal==1.6.4
modal setup
modal volume create mammo-input
modal volume create mammo-resultados
modal volume put mammo-input ./desenvolvimento /desenvolvimento
modal volume ls mammo-input /desenvolvimento
modal run sprints/modal_app.py --sprint 7 --mode smoke
```

O resultado haverá de imprimir o comando `modal volume get`. Modal usa o login do próprio CLI, não chave Kaggle; não cole tokens em células ou no Git. Para executar outra Sprint, altere `--sprint`; para S07–S09 experimentais use `--mode experiment`; para escolher braço S09 use `--arm batch-16-fixed` ou outro permitido.

## 3. Primeiro smoke, depois protocolo

1. Abra a S07 e rode com `SMOKE=True` (Modal: `--mode smoke`). Faz uma época, batch 4, até 100 imagens (até 25 por classe em cada split) e sem download de pesos. Confira `summary.json`, um JSON de predições de validação e um checkpoint. **Não compare o AUROC do smoke aos slides.**
2. Se funcionou, use `SMOKE=False` nas S07–S09 (Modal: `--mode experiment`). Isso usa os parâmetros do TOML. S09 não dispara a grade toda ao clicar em Run All.
3. Use S10/S11 somente depois que os CSVs reais exigidos estiverem disponíveis. Não haverá números de teste final nesses cadernos.

Erros comuns: `manifest.csv` ausente → confira montagem e `DATA`; `CUDA indisponível` → troque o runtime para GPU e reinicie; `torchvision` incompatível → reinicie no ambiente padrão (não atualize só um dos dois pacotes); erro ao decodificar DICOM → habilite internet para os plugins ou prepare imagens PNG autorizadas; OOM → não trate um batch menor como o braço original. Os cadernos não pedem credenciais nem registram pesos/outputs em repositório.
