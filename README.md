# classificacaofakenews

Para treinar os classificadores rode: classify.py
Para testar os modelos rode: test_classifier.py

Obs.: É importante editar editar o caminho onde os dados serão lidos (input_path).
Também altere a versão do dataset (variável dataset_version). Utilize "v2" para rodar os embeddings de dimensão 100, e "v3" para o de dimensão 300. 

Os parâmetros da leitura dessas versões podem ser vistos no método get_dataset_params do arquivo reader.py

Dataset original:
https://github.com/entitize/fakeddit

Dataset processado:
https://drive.google.com/drive/folders/1gbiXD5ngBaJt8CpfyiGrKpXg-V95rJTX?usp=sharing

Para gerar o repositório com word embedding de tamanho 100 rode: generate_dataset.py

Observação: O dataset processado está com todos os registros do original, mas foi incluída uma coluna com o número de palavras da notícia, para que seja possível filtrar por meio da aplicação. Neste trabalho ele foi filtrado para excluir palavras de tamanho menor que 6, diretamente no código de leitura reader.py
