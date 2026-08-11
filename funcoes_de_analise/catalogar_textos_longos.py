import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Carregamento preguiçoso do modelo e cache do corpus lematizado (reaproveitado entre
# chamadas com os mesmos textos, ex.: ao pontuar vários temas sobre o mesmo corpus).
_nlp = None
_cache_corpus_lematizado = {}


def _get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load("pt_core_news_lg", disable=["parser", "ner"])
    return _nlp


def _lematizar_lista(textos):
    """Lematiza uma lista de textos (minúsculas, só tokens alfabéticos). Usa cache."""
    chave = hash(tuple(textos))
    if chave in _cache_corpus_lematizado:
        return _cache_corpus_lematizado[chave]

    nlp = _get_nlp()
    lematizados = []
    total = len(textos)
    for i, doc in enumerate(nlp.pipe([str(t) for t in textos], batch_size=50)):
        lematizados.append(" ".join(tok.lemma_.lower() for tok in doc if tok.is_alpha))
        if total > 0 and (i + 1) % max(1, total // 100) == 0:
            pct = (i + 1) / total * 100
            print(f"\rLematizando corpus: {pct:.1f}% [{i + 1}/{total}]", end="", flush=True)
    if total > 0:
        print()

    _cache_corpus_lematizado[chave] = lematizados
    return lematizados


def _lematizar_termo(palavra):
    """Retorna o lema de uma palavra do conjunto de busca."""
    nlp = _get_nlp()
    for tok in nlp(str(palavra).lower()):
        if tok.is_alpha:
            return tok.lemma_.lower()
    return str(palavra).lower()


def lematizar_conjunto(palavras):
    """
    Lematiza um conjunto/lista de palavras, retornando um set de lemas.

    Útil para preparar o set de busca na mesma forma do corpus lematizado, deixando a
    lematização explícita no ponto da chamada:
        catalogar_textos_longos(titulos, textos, lematizar_conjunto({"crianças"}), "Família")

    Args:
        palavras (iterable of str): Palavras do assunto (em qualquer forma flexionada).

    Returns:
        set of str: Conjunto das palavras já lematizadas (ex.: {"crianças"} -> {"criança"}).
    """
    return {_lematizar_termo(p) for p in palavras}


def catalogar_textos_longos(titulos, textos, set_de_palavras, nome_do_conjunto, lematizar=True):
    """
    Pontua textos por assunto usando TF-IDF (Frequência do Termo - Inverso da Frequência
    nos Documentos).

    A função monta a matriz TF-IDF do corpus e, para cada texto, soma os pesos TF-IDF das
    palavras do conjunto `set_de_palavras` (o "assunto"). Essa soma é a grandeza atribuída
    ao texto: quanto maior, mais o texto corresponde ao assunto buscado. Como o TF-IDF
    normaliza cada documento (norma L2), a grandeza é comparável entre textos de tamanhos
    diferentes. O resultado é ordenado do texto mais aderente ao menos aderente.

    Args:
        titulos (list of str): Identificadores dos textos (ex.: títulos). Usados como
            chaves do dicionário de saída.
        textos (list of str): Os textos, na mesma ordem de `titulos`.
        set_de_palavras (set of str): Palavras que definem o assunto a ser buscado.
        nome_do_conjunto (str): Nome do conjunto/assunto. Vira a chave interna que guarda
            a grandeza de cada texto.
        lematizar (bool): Se True (padrão), o corpus e as palavras do conjunto são
            lematizados com o spaCy antes do TF-IDF, de modo que formas flexionadas contem
            juntas ("orações" casa com "oração"). O corpus lematizado é cacheado e
            reaproveitado entre chamadas com os mesmos textos.

    Returns:
        dict: Dicionário ordenado do texto mais aderente ao menos aderente, no formato:
            { titulo: { nome_do_conjunto: grandeza }, ... }
        onde `grandeza` (float) é a soma dos pesos TF-IDF das palavras do conjunto no texto.
    """
    if len(titulos) != len(textos):
        raise ValueError(
            f"titulos ({len(titulos)}) e textos ({len(textos)}) devem ter o mesmo tamanho."
        )

    # 0. Lematização (opcional) do corpus e das palavras do conjunto
    if lematizar:
        corpus = _lematizar_lista(list(textos))
        palavras = {_lematizar_termo(p) for p in set_de_palavras}
    else:
        corpus = [str(t) for t in textos]
        palavras = {str(p).lower() for p in set_de_palavras}

    # 1. Matriz TF-IDF do corpus (cada linha = um texto, normalizada em L2)
    vectorizer = TfidfVectorizer(lowercase=True)
    tfidf = vectorizer.fit_transform(corpus)
    vocabulario = vectorizer.vocabulary_

    # 2. Colunas correspondentes às palavras do conjunto presentes no vocabulário
    colunas = [vocabulario[p] for p in palavras if p in vocabulario]

    # 3. Grandeza de cada texto = soma dos pesos TF-IDF das palavras do conjunto
    if colunas:
        grandezas = np.asarray(tfidf[:, colunas].sum(axis=1)).ravel()
    else:
        grandezas = np.zeros(len(textos))

    # 4. Ordena do mais aderente ao menos aderente e monta o dicionário de saída
    ordem = np.argsort(grandezas)[::-1]
    resultado = {}
    for i in ordem:
        resultado[titulos[i]] = {nome_do_conjunto: float(grandezas[i])}

    return resultado
