# EP1 - Oblique Random Forest

## Bibliotecas necessárias

Instale as dependências com:

```bash
pip install numpy pandas scikit-learn
```

Bibliotecas usadas no código:

- `numpy`
- `pandas`
- `scikit-learn`

## Como rodar

O programa recebe um arquivo de treino e um arquivo de teste em formato CSV.

Exemplo:

```bash
python ep_orf.py --train_csv exemplo_train.csv --test_csv exemplo_test.csv --target target --id_column id --submission submission.csv
```

## Parâmetros principais

- `--train_csv`: arquivo CSV de treino.
- `--test_csv`: arquivo CSV de teste.
- `--target`: nome da coluna que contém a classe correta no treino.
- `--id_column`: nome da coluna de identificação das amostras.
- `--submission`: nome do arquivo CSV de saída.

## Exemplo com arquivos grandes

```bash
python ep_orf.py --train_csv train.csv --test_csv test.csv --target target --id_column id --submission submission.csv
```

## Saída

O código imprime no terminal a comparação entre a oRF implementada e uma Random Forest tradicional, usando uma validação interna do arquivo de treino e é gerado o arquivo de saída `--submission`.
