import numpy as np

# Carregamento preguiçoso do modelo com vetores (pt_core_news_lg tem word vectors de 300 dim)
_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load("pt_core_news_lg", disable=["parser", "ner"])
    return _nlp


def encontrar_palavras_similares(palavras, sentencas, limiar=0.4, lemma=True,
                                 min_tamanho=3, max_sentencas=None, incluir_similaridade=False):
    """
    Encontra palavras do corpus semanticamente próximas a um conjunto de palavras-semente
    e devolve, para cada uma, as sentenças em que aparece.

    Para cada palavra do corpus, calcula a menor DISTÂNCIA de cosseno (1 - similaridade)
    até as palavras-semente. As palavras cuja distância fica ABAIXO do `limiar` são
    consideradas próximas e retornadas, ordenadas da mais próxima para a mais distante.

    Args:
        palavras (list of str): Palavras-semente (definem o eixo semântico buscado).
        sentencas (list of str): Corpus segmentado em sentenças.
        limiar (float): Distância de cosseno máxima (0 = idêntico; menor = mais parecido).
            Palavras com distância <= limiar são retornadas. Valores típicos: 0.3 a 0.6.
            Padrão: 0.4.
        lemma (bool): Se True, agrupa e compara as palavras por lema. Padrão: True.
        min_tamanho (int): Ignora palavras com menos de `min_tamanho` caracteres. Padrão: 3.
        max_sentencas (int | None): Se informado, limita quantas sentenças são guardadas
            por palavra encontrada. Padrão: None (todas).
        incluir_similaridade (bool): Se True, cada valor passa a ser um dict com a
            similaridade, a distância e as sentenças. Se False (padrão), o valor é apenas
            a lista de sentenças.

    Returns:
        list of dict: Lista de dicionários de uma chave cada, ordenada da palavra mais
        próxima (semanticamente) para a mais distante. As palavras-semente são excluídas.
        - Se `incluir_similaridade=False` (padrão):
            [{palavra: [sentenças...]}, ...]
        - Se `incluir_similaridade=True`:
            [{palavra: {"similaridade": float, "distancia": float, "sentencas": [...]}}, ...]
    """
    nlp = _get_nlp()

    def _vetor(chave):
        lex = nlp.vocab[chave]
        return lex.vector if (lex.has_vector and lex.vector_norm > 0) else None

    # 1. Vetores das sementes
    sementes, vetores_sementes = set(), []
    for p in palavras:
        doc = nlp(str(p).lower())
        if not len(doc):
            continue
        chave = doc[0].lemma_.lower() if lemma else doc[0].text.lower()
        v = _vetor(chave)
        if v is not None:
            sementes.add(chave)
            vetores_sementes.append(v)
    if not vetores_sementes:
        print("Nenhuma palavra-semente possui vetor no modelo.")
        return []

    M_sem = np.array(vetores_sementes)
    M_sem = M_sem / np.linalg.norm(M_sem, axis=1, keepdims=True)

    # 2. Varre o corpus: coleta candidatos (vetor) e as sentenças de cada palavra
    vetor_por_palavra = {}
    sentencas_por_palavra = {}
    total = len(sentencas)
    for i, doc in enumerate(nlp.pipe(sentencas, batch_size=50)):
        vistos = set()
        for tok in doc:
            if not tok.is_alpha or tok.is_stop or len(tok) < min_tamanho:
                continue
            chave = tok.lemma_.lower() if lemma else tok.text.lower()
            if chave in sementes:
                continue

            # Guarda o vetor da palavra na primeira vez que a vê (ignora quem não tem vetor)
            if chave not in vetor_por_palavra:
                v = _vetor(chave)
                if v is None:
                    continue
                vetor_por_palavra[chave] = v

            # Registra a sentença (uma vez por sentença) na lista da palavra
            if chave not in vistos:
                lst = sentencas_por_palavra.setdefault(chave, [])
                if max_sentencas is None or len(lst) < max_sentencas:
                    lst.append(doc.text)
                vistos.add(chave)
        if total > 0 and (i + 1) % max(1, total // 100) == 0:
            print(f"\rProcessando sentenças: {(i + 1) / total * 100:.1f}% [{i + 1}/{total}]",
                  end="", flush=True)
    if total > 0:
        print()

    if not vetor_por_palavra:
        return []

    # 3. Distância de cosseno de cada candidato à semente mais próxima
    candidatos = list(vetor_por_palavra.keys())
    M_cand = np.array([vetor_por_palavra[c] for c in candidatos])
    M_cand = M_cand / np.linalg.norm(M_cand, axis=1, keepdims=True)
    similaridades = M_cand @ M_sem.T            # (n_candidatos x n_sementes)
    distancias = 1 - similaridades.max(axis=1)  # distância à semente mais próxima

    # 4. Filtra por limiar e ordena da mais próxima para a mais distante
    resultado = []
    for idx in np.argsort(distancias):
        if distancias[idx] <= limiar:
            palavra = candidatos[idx]
            sents = sentencas_por_palavra[palavra]
            if incluir_similaridade:
                resultado.append({palavra: {
                    "similaridade": float(1 - distancias[idx]),
                    "distancia": float(distancias[idx]),
                    "sentencas": sents,
                }})
            else:
                resultado.append({palavra: sents})

    return resultado
