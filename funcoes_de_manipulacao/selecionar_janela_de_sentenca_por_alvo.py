import re


def selecionar_janela_de_sentenca_por_alvo(lista_sentencas, set_de_alvos, janela=1):
    """
    Seleciona sentenças que contenham algum dos termos-alvo, agrupando ocorrências próximas.

    A função varre a lista de sentenças e identifica aquelas em que aparece pelo menos
    um dos alvos (correspondência por palavra inteira, ignorando maiúsculas/minúsculas
    e pontuação). Quando duas sentenças-alvo estão separadas por no máximo `janela`
    sentenças sem alvo, elas são consideradas parte do mesmo trecho e unidas em uma
    única string (juntas por ", "). As sentenças intermediárias, sem alvo, são descartadas.

    Exemplo:
        alvos = {"deus", "senhor"}
        sentencas = [
            "Que deus me ajude a achar meu filho",   # contém alvo (índice 0)
            "ele se perdeu na mata",                  # sem alvo  (índice 1)
            "agradeço ao senhor por sua vida",        # contém alvo (índice 2)
        ]
        selecionar_sentenca_por_alvo(sentencas, alvos, janela=1)
        # -> ["Que deus me ajude a achar meu filho, agradeço ao senhor por sua vida"]

    Args:
        lista_sentencas (list of str): Corpus segmentado em sentenças.
        set_de_alvos (set of str): Conjunto de termos-alvo. É normalizado para minúsculas.
        janela (int): Número máximo de sentenças sem alvo permitido entre duas
            sentenças-alvo para que elas sejam agrupadas. Padrão: 1.
            Com janela=0, apenas sentenças-alvo consecutivas são unidas.

    Returns:
        list of str: Lista de trechos. Cada trecho é a junção das sentenças-alvo de um
        mesmo grupo. Retorna lista vazia se nenhuma sentença contiver um alvo.
    """
    set_de_alvos = {alvo.lower() for alvo in set_de_alvos}

    # 1. Identifica os índices das sentenças que contêm algum alvo
    indices_alvo = []
    for i, sent in enumerate(lista_sentencas):
        palavras_sentenca = set(re.findall(r'\b\w+\b', sent.lower()))
        if set_de_alvos.intersection(palavras_sentenca):
            indices_alvo.append(i)

    if not indices_alvo:
        return []

    # 2. Agrupa índices-alvo separados por no máximo `janela` sentenças sem alvo
    grupos = []
    grupo_atual = [indices_alvo[0]]
    for idx in indices_alvo[1:]:
        if idx - grupo_atual[-1] <= janela + 1:
            grupo_atual.append(idx)
        else:
            grupos.append(grupo_atual)
            grupo_atual = [idx]
    grupos.append(grupo_atual)

    # 3. Junta as sentenças-alvo de cada grupo em uma única string
    saida = [", ".join(lista_sentencas[i] for i in grupo) for grupo in grupos]
    return saida
