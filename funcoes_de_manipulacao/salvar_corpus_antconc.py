import os
import re
import unicodedata


def _sanitizar_nome_arquivo(valor):
    """
    Normaliza um valor para uso seguro em nome de arquivo.

    Remove acentos, troca espaços por underscore e descarta qualquer caractere que
    não seja letra, número, '_' ou '-' (evita caracteres inválidos no Windows, como
    < > : " / \\ | ? *).
    """
    valor = str(valor).strip()
    # Remove acentos (ç -> c, ã -> a, etc.)
    valor = unicodedata.normalize("NFKD", valor)
    valor = "".join(c for c in valor if not unicodedata.combining(c))
    valor = valor.replace(" ", "_")
    # Mantém apenas caracteres seguros para nome de arquivo
    valor = re.sub(r"[^A-Za-z0-9_-]+", "", valor)
    return valor.strip("_-")


def salvar_corpus_antconc(lista_textos, caminho_pasta, lista_metadados=None):
    """
    Salva um corpus para o software AntConc: um arquivo .txt por texto, em UTF-8.

    O AntConc trata cada arquivo como um texto independente (essencial para as
    estatísticas de File/Range). Como o AntConc não possui campo de metadados, as
    informações de cada texto são codificadas no NOME do arquivo. Cada nome recebe
    um índice numérico no início para garantir ordem e unicidade.

    Estrutura gerada (exemplo, com metadados de data e canal):
        corpus_antconc/
            0001_2020-03-15_nikolas.txt
            0002_2020-04-02_nikolas.txt
            ...

    Args:
        lista_textos (list of str): Lista de textos (cada elemento vira um arquivo).
        caminho_pasta (str): Pasta de destino. É criada se não existir.
        lista_metadados (list of dict, opcional): Lista paralela a `lista_textos` com
            os metadados de cada texto. Apenas os VALORES são usados (na ordem do dict)
            para compor o nome do arquivo, sanitizados (sem acentos/espaços/caracteres
            inválidos). Ex.: [{"data": "2020-03-15", "canal": "Nikolas"}, ...] gera
            "0001_2020-03-15_Nikolas.txt". Se None, usa apenas o índice: "0001.txt".

    Returns:
        list of str: Lista dos caminhos dos arquivos efetivamente gravados.

    Observações:
        - Salva em UTF-8 sem BOM (padrão esperado pelo AntConc).
        - Textos vazios (após strip) são ignorados.
        - Se `lista_metadados` for fornecida, deve ter o mesmo tamanho de `lista_textos`.
    """
    if lista_metadados is not None and len(lista_metadados) != len(lista_textos):
        raise ValueError(
            f"lista_metadados ({len(lista_metadados)}) deve ter o mesmo tamanho de "
            f"lista_textos ({len(lista_textos)})."
        )

    os.makedirs(caminho_pasta, exist_ok=True)

    largura_indice = max(4, len(str(len(lista_textos))))
    caminhos_gravados = []

    for i, texto in enumerate(lista_textos):
        conteudo = str(texto).strip()
        if not conteudo:
            continue  # ignora textos vazios

        # Monta o nome do arquivo: índice + valores dos metadados
        partes = [f"{i + 1:0{largura_indice}d}"]
        if lista_metadados is not None:
            for valor in lista_metadados[i].values():
                parte = _sanitizar_nome_arquivo(valor)
                if parte:
                    partes.append(parte)

        nome_arquivo = "_".join(partes) + ".txt"
        caminho_destino = os.path.join(caminho_pasta, nome_arquivo)

        with open(caminho_destino, "w", encoding="utf-8") as f:
            f.write(conteudo)

        caminhos_gravados.append(caminho_destino)

    print(f"Corpus AntConc gerado em: {caminho_pasta}")
    print(f" - Arquivos escritos: {len(caminhos_gravados)} (de {len(lista_textos)} recebidos)")

    return caminhos_gravados
