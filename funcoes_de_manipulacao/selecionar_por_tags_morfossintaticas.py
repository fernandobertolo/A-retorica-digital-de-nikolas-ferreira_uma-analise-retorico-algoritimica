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
        list of str | None: Lista das orações (sem duplicatas, na ordem original) em que o
        termo ocorre com as tags especificadas. Retorna None se nada casar.

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

    # Retorna None quando nada foi encontrado (em vez de lista vazia)
    return sentencas_selecionadas if sentencas_selecionadas else None


def _token_casa_tags(token, tag_set):
    """Um token casa o tag_set se TODAS as tags do set forem seu POS ou seu DEP.
    tag_set None (ou vazio) = qualquer tag."""
    if not tag_set:
        return True
    return tag_set.issubset({token.pos_.upper(), token.dep_.upper()})


def selecionar_frase_por_conjunto_de_termos_e_tags(termos_e_tags, lista_de_sentencas, lemma=True):
    """
    Seleciona as frases que satisfazem simultaneamente um conjunto de requisitos de
    (termo, tags morfossintáticas).

    Cada requisito é uma tupla `(termo, {tags}, termo_pai)`, interpretada assim:
      - ("palavra", {tags}) -> a frase deve conter "palavra" exercendo TODAS essas tags
        (cada tag deve ser o POS ou o DEP do token; ex.: {"NOUN", "nsubj"} = substantivo E
        sujeito).
      - ("palavra",)  ou  ("palavra", None)  ->  a frase deve conter "palavra", com qualquer tag.
      - ("", {tags})  ->  a frase deve conter ALGUM termo (qualquer) que exerça essas tags.
      - ("", {tags}, "pai") -> além das tags, o token precisa ser SUBORDINADO (filho na
        árvore de dependências do spaCy) a um token cujo termo seja "pai". Ou seja,
        token.head deve ser o termo "pai" (que, se declarado em outra tupla, também deve
        satisfazer as tags daquela tupla).

    A frase é selecionada apenas se atender a TODAS as tuplas da lista (E lógico).

    Exemplos:
        # frases em que "deus" é sujeito E que também contenham algum verbo
        selecionar_frase_por_conjunto_de_termos_e_tags(
            [("deus", {"nsubj"}), ("", {"VERB"})], sentencas)

        # frases em que "deus" é PROPN E possui um determinante subordinado (ex.: "o nosso deus")
        selecionar_frase_por_conjunto_de_termos_e_tags(
            [("deus", {"PROPN"}), ("", {"DET", "det"}, "deus")], sentencas)

    Args:
        termos_e_tags (list of tuple): Lista de requisitos, cada um uma tupla
            (termo, set_de_tags[, termo_pai]) conforme descrito acima. O terceiro valor
            (opcional) é o termo ao qual o token deve estar subordinado na árvore de
            dependências.
        lista_de_sentencas (list of str): Corpus segmentado em sentenças.
        lemma (bool): Se True, compara os termos e os tokens por sua forma lematizada.
            Padrão: True.

    Returns:
        list of str | None: Frases (na ordem original) que satisfazem todas as tuplas.
        Retorna None se nenhuma frase atender ou se `termos_e_tags` estiver vazia.

    Observações:
        - As tags seguem o padrão Universal POS / dependências do spaCy (comparação sem
          diferenciar maiúsculas/minúsculas).
        - Como cada token tem um único POS e um único DEP, um set com dois POS (ex.:
          {"NOUN", "VERB"}) nunca casa; use tuplas separadas se precisar de alternativas.
    """
    if not termos_e_tags:
        return None

    def _lematizar_termo(termo):
        """Normaliza um termo: '' -> None (curinga); senão lematiza (se lemma) e minusculiza."""
        termo = (termo or "").strip()
        if termo == "":
            return None
        alvo = termo.lower()
        if lemma:
            doc_alvo = nlp(alvo)
            alvo = doc_alvo[0].lemma_.lower() if len(doc_alvo) > 0 else alvo
        return alvo

    def _forma(token):
        return token.lemma_.lower() if lemma else token.text.lower()

    # Pré-processa cada tupla -> (alvo, tag_set, pai_alvo)
    requisitos = []
    tags_por_termo = {}  # termo -> tag_set, para validar as tags do "pai" na árvore
    for tupla in termos_e_tags:
        alvo = _lematizar_termo(tupla[0])

        tag_set = _normalizar_tags(tupla[1] if len(tupla) > 1 else None)
        if tag_set is not None and len(tag_set) == 0:
            tag_set = None  # set vazio = qualquer tag

        pai_alvo = _lematizar_termo(tupla[2]) if len(tupla) > 2 else None

        requisitos.append((alvo, tag_set, pai_alvo))
        if alvo is not None:
            tags_por_termo[alvo] = tag_set

    def _requisito_satisfeito(doc, alvo, tag_set, pai_alvo):
        for token in doc:
            # Casa o próprio token (termo + tags)
            if alvo is not None and _forma(token) != alvo:
                continue
            if not _token_casa_tags(token, tag_set):
                continue

            # Casa a subordinação na árvore de dependências (token.head == termo "pai")
            if pai_alvo is not None:
                cabeca = token.head
                if cabeca is token:  # token raiz não tem um "pai" de verdade
                    continue
                if _forma(cabeca) != pai_alvo:
                    continue
                # Se o "pai" também é declarado como tupla, sua tags precisa bater no head
                if not _token_casa_tags(cabeca, tags_por_termo.get(pai_alvo)):
                    continue

            return True
        return False

    sentencas_selecionadas = []
    total_sentencas = len(lista_de_sentencas)

    for idx, doc in enumerate(nlp.pipe(lista_de_sentencas)):
        # Marcador de progresso
        if total_sentencas > 0 and (idx + 1) % max(1, total_sentencas // 100) == 0:
            progresso = ((idx + 1) / total_sentencas) * 100
            print(f"\rProcessando sentenças: {progresso:.1f}% concluído [{idx + 1}/{total_sentencas}]",
                  end="", flush=True)

        # A frase precisa satisfazer TODOS os requisitos (tuplas)
        if all(_requisito_satisfeito(doc, alvo, tag_set, pai_alvo)
               for alvo, tag_set, pai_alvo in requisitos):
            sentencas_selecionadas.append(doc.text)

    if total_sentencas > 0:
        print("\n")  # quebra de linha após a barra de progresso

    return sentencas_selecionadas if sentencas_selecionadas else None
