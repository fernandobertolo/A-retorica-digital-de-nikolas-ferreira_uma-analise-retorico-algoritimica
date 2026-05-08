import json, os


def carregar_json_para_lista(caminho_arquivo):
    """
    Lê um arquivo JSON e retorna uma lista de dicionários.
    """
    if not os.path.exists(caminho_arquivo):
        raise FileNotFoundError(f"O arquivo não foi encontrado em: {caminho_arquivo}")

    with open(caminho_arquivo, "r", encoding="utf-8") as arquivo:
        dados = json.load(arquivo)

    if not isinstance(dados, list):
        raise TypeError("O conteúdo do JSON não é uma lista.")

    return dados