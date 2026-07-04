import re


def selecionar_janela_de_sentenca_por_alvo(lista_sentencas, set_de_alvos, janela=1, lematizar=False):
    """
    Seleciona sentenças que contenham algum dos termos-alvo, agrupando ocorrências próximas.

    A função varre a lista de sentenças e identifica aquelas em que aparece pelo menos
    um dos alvos. Quando duas sentenças-alvo estão separadas por no máximo `janela`
    sentenças sem alvo, elas são consideradas parte do mesmo trecho e unidas em uma
    única string (juntas por ", "). As sentenças intermediárias, sem alvo, são descartadas.

    A correspondência pode ser feita de duas formas:
      - `lematizar=False` (padrão): forma de superfície exata, por palavra inteira e
        ignorando maiúsculas/minúsculas. Rápido (apenas regex + operações de conjunto),
        mas "deus" não casa com "deuses".
      - `lematizar=True`: compara os LEMAS dos tokens e dos alvos usando o spaCy, o que
        permite casar formas flexionadas ("deuses" casa com "deus"; "orou" com "orar").
        Bem mais custoso, pois carrega e executa o modelo de linguagem.

    Exemplo:
        alvos = {"deus", "senhor"}
        sentencas = [
            "Que deus me ajude a achar meu filho",   # contém alvo (índice 0)
            "ele se perdeu na mata",                  # sem alvo  (índice 1)
            "agradeço ao senhor por sua vida",        # contém alvo (índice 2)
        ]
        selecionar_janela_de_sentenca_por_alvo(sentencas, alvos, janela=1)
        # -> ["Que deus me ajude a achar meu filho, agradeço ao senhor por sua vida"]

    Args:
        lista_sentencas (list of str): Corpus segmentado em sentenças.
        set_de_alvos (set of str): Conjunto de termos-alvo. É normalizado para minúsculas.
        janela (int): Número máximo de sentenças sem alvo permitido entre duas
            sentenças-alvo para que elas sejam agrupadas. Padrão: 1.
            Com janela=0, apenas sentenças-alvo consecutivas são unidas.
        lematizar (bool): Se True, a busca considera o lema (lexema) dos termos e dos
            tokens das sentenças, casando formas flexionadas. Usa o spaCy
            (pt_core_news_lg) e é bem mais lento. Se False (padrão), busca literal
            por forma de superfície com regex. Padrão: False.

    Returns:
        list of str: Lista de trechos. Cada trecho é a junção das sentenças-alvo de um
        mesmo grupo (sempre no texto ORIGINAL da sentença). Retorna lista vazia se
        nenhuma sentença contiver um alvo.
    """
    # 1. Identifica os índices das sentenças que contêm algum alvo
    indices_alvo = []
    total_sentencas = len(lista_sentencas)

    def _progresso(idx):
        if total_sentencas > 0 and (idx + 1) % max(1, total_sentencas // 100) == 0:
            pct = ((idx + 1) / total_sentencas) * 100
            print(f"\rProcessando sentenças: {pct:.1f}% concluído [{idx + 1}/{total_sentencas}]",
                  end="", flush=True)

    if lematizar:
        # Busca por lema: lematiza os alvos e os tokens de cada sentença com o spaCy.
        import spacy
        nlp = spacy.load("pt_core_news_lg", disable=["parser", "ner"])

        alvos_lema = set()
        for alvo in set_de_alvos:
            for token in nlp(alvo):
                if token.is_alpha:
                    alvos_lema.add(token.lemma_.lower())

        for i, doc in enumerate(nlp.pipe(lista_sentencas, batch_size=50)):
            lemas_sentenca = {token.lemma_.lower() for token in doc if token.is_alpha}
            if alvos_lema.intersection(lemas_sentenca):
                indices_alvo.append(i)
            _progresso(i)
    else:
        # Busca literal (forma de superfície), rápida, apenas com regex.
        set_de_alvos = {alvo.lower() for alvo in set_de_alvos}
        for i, sent in enumerate(lista_sentencas):
            palavras_sentenca = set(re.findall(r'\b\w+\b', sent.lower()))
            if set_de_alvos.intersection(palavras_sentenca):
                indices_alvo.append(i)
            _progresso(i)

    if total_sentencas > 0:
        print()  # quebra de linha após a barra de progresso

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
