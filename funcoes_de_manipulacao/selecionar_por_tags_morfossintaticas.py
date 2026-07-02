import spacy

# Carregamento do modelo linguístico
try:
    nlp = spacy.load("pt_core_news_sm")
except OSError:
    raise OSError("Modelo linguístico não encontrado. Execute: python -m spacy download pt_core_news_sm")


def _normalizar_tags(tags):
    """Converte a entrada (None, str ou iterável) em um set de tags em maiúsculas, ou None."""
    if tags is None:
        return None
    if isinstance(tags, str):
        tags = [tags]
    return {t.upper() for t in tags}


def selecionar_por_tags_morfossintaticas(termo, lista_de_sentencas, pos=None, dep=None, lemma=True):
    """
    Seleciona as orações em que um termo aparece exercendo determinada função morfossintática.

    A função varre a lista de sentenças procurando o `termo` e, quando o encontra, verifica
    a classe gramatical (POS) e/ou o papel sintático (dependência, DEP) que ele exerce
    naquela ocorrência (via parser do spaCy). A oração é selecionada apenas se a ocorrência
    casar com os filtros de `pos` e `dep` informados.

    Exemplo:
        sentencas = [
            "O estado forte garante os direitos.",   # 'estado' = NOUN
            "Eles vão estado alerta o dia todo.",     # (hipotético) 'estado' com outra função
        ]
        selecionar_por_tags_morfossintaticas("estado", sentencas, pos="NOUN")
        # -> ["O estado forte garante os direitos."]

    Args:
        termo (str): Termo-alvo a ser buscado nas sentenças.
        lista_de_sentencas (list of str): Corpus segmentado em sentenças.
        pos (str | iterable of str | None): Classe(s) gramatical(is) aceita(s)
            (ex.: "VERB" ou {"NOUN", "ã"}). Se None, aceita qualquer POS.
        dep (str | iterable of str | None): Papel(is) sintático(s) aceito(s)
            (ex.: "obj" ou {"nsubj", "obj"}). Se None, aceita qualquer dependência.
        lemma (bool): Se True, compara o termo e os tokens por sua forma lematizada
            (permite passar o termo flexionado). Padrão: True.

    Returns:
        list of str: Lista das orações (sem duplicatas, na ordem original) em que o termo
        ocorre com as tags especificadas. Retorna lista vazia se nada casar.

    Observações:
        - As tags seguem o padrão Universal POS / dependências do spaCy.
        - Uma oração entra na saída uma única vez, mesmo que o termo apareça mais de uma
          vez nela com a tag desejada.
    """
    alvo = termo.lower()
    if lemma:
        doc_alvo = nlp(alvo)
        alvo = doc_alvo[0].lemma_.lower() if len(doc_alvo) > 0 else alvo

    pos_filtro = _normalizar_tags(pos)
    dep_filtro = _normalizar_tags(dep)

    sentencas_selecionadas = []
    total_sentencas = len(lista_de_sentencas)

    for idx, doc in enumerate(nlp.pipe(lista_de_sentencas)):
        # Marcador de progresso
        if total_sentencas > 0 and (idx + 1) % max(1, total_sentencas // 100) == 0:
            progresso = ((idx + 1) / total_sentencas) * 100
            print(f"\rProcessando sentenças: {progresso:.1f}% concluído [{idx + 1}/{total_sentencas}]",
                  end="", flush=True)

        for token in doc:
            forma_analisada = token.lemma_.lower() if lemma else token.text.lower()

            if forma_analisada != alvo:
                continue

            # Verifica os filtros de POS e de dependência (DEP)
            casa_pos = pos_filtro is None or token.pos_.upper() in pos_filtro
            casa_dep = dep_filtro is None or token.dep_.upper() in dep_filtro

            if casa_pos and casa_dep:
                sentencas_selecionadas.append(doc.text)
                break  # basta uma ocorrência válida por oração

    if total_sentencas > 0:
        print("\n")  # quebra de linha após a barra de progresso

    return sentencas_selecionadas
