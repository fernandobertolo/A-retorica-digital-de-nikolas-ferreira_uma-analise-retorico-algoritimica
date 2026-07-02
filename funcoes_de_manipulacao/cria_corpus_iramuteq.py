import re
import unicodedata


def _sanitizar_valor_variavel(valor):
    """
    Normaliza o valor de uma variável estrelada para o formato aceito pelo Iramuteq.

    Remove acentos, converte para minúsculas e substitui qualquer caractere que não
    seja letra/número por underscore (as modalidades não podem conter espaços, '*'
    ou caracteres especiais).
    """
    valor = str(valor).strip().lower()
    # Remove acentos (ç -> c, ã -> a, etc.)
    valor = unicodedata.normalize("NFKD", valor)
    valor = "".join(c for c in valor if not unicodedata.combining(c))
    # Tudo que não for alfanumérico vira underscore
    valor = re.sub(r"[^a-z0-9]+", "_", valor)
    valor = valor.strip("_")
    return valor if valor else "na"


def _limpar_texto(texto):
    """
    Prepara o corpo do texto para o Iramuteq.

    Remove o caractere '*' (reservado para as linhas de comando), colapsa quebras de
    linha e espaços múltiplos em um único espaço e remove espaços nas pontas.
    """
    texto = str(texto).replace("*", " ")
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip()


def cria_corpus_iramuteq(lista_textos,
                         caminho_saida="corpus_iramuteq.txt",
                         lista_metadados=None,
                         nome_variavel="texto"):
    """
    Gera um arquivo de corpus no formato do software Iramuteq a partir de uma lista de textos.

    Cada texto vira um "texto" do corpus, precedido por uma linha de comando estrelada
    no formato `**** *variavel_valor`. O arquivo é salvo em UTF-8, que é o padrão
    esperado pelo Iramuteq.

    Estrutura gerada (exemplo):
        **** *texto_0001
        conteudo do primeiro texto ...

        **** *texto_0002
        conteudo do segundo texto ...

    Args:
        lista_textos (list of str): Lista de textos (cada elemento é um texto do corpus).
        caminho_saida (str): Caminho do arquivo .txt de saída. Padrão: "corpus_iramuteq.txt".
        lista_metadados (list of dict, opcional): Lista paralela a `lista_textos` com as
            variáveis estreladas de cada texto, no formato {nome_variavel: valor}.
            Ex.: [{"canal": "nikolas", "ano": 2020}, ...]. Nomes e valores são
            normalizados (sem acentos/espaços). Se None, é gerada automaticamente uma
            variável índice `*{nome_variavel}_NNNN`.
        nome_variavel (str): Nome da variável índice usada quando `lista_metadados`
            é None. Padrão: "texto".

    Returns:
        str: O caminho do arquivo de corpus gerado.

    Observações:
        - O caractere '*' é removido do corpo dos textos (reservado para as linhas de comando).
        - Textos vazios (após limpeza) são ignorados.
        - Se `lista_metadados` for fornecida, deve ter o mesmo tamanho de `lista_textos`.
    """
    if lista_metadados is not None and len(lista_metadados) != len(lista_textos):
        raise ValueError(
            f"lista_metadados ({len(lista_metadados)}) deve ter o mesmo tamanho de "
            f"lista_textos ({len(lista_textos)})."
        )

    nome_variavel = _sanitizar_valor_variavel(nome_variavel)
    blocos = []
    total_caracteres = 0

    for i, texto in enumerate(lista_textos):
        corpo = _limpar_texto(texto)
        if not corpo:
            continue  # ignora textos vazios

        # Monta a linha de comando estrelada
        if lista_metadados is not None:
            variaveis = " ".join(
                f"*{_sanitizar_valor_variavel(chave)}_{_sanitizar_valor_variavel(valor)}"
                for chave, valor in lista_metadados[i].items()
            )
            linha_comando = f"**** {variaveis}"
        else:
            linha_comando = f"**** *{nome_variavel}_{i + 1:04d}"

        blocos.append(f"{linha_comando}\n{corpo}")
        total_caracteres += len(corpo)

    # Iramuteq separa os textos por uma linha em branco
    conteudo = "\n\n".join(blocos) + "\n"

    with open(caminho_saida, "w", encoding="utf-8") as arquivo:
        arquivo.write(conteudo)

    print(f"Corpus Iramuteq gerado: {caminho_saida}")
    print(f" - Textos escritos     : {len(blocos)} (de {len(lista_textos)} recebidos)")
    print(f" - Caracteres no corpo : {total_caracteres}")

    return caminho_saida
