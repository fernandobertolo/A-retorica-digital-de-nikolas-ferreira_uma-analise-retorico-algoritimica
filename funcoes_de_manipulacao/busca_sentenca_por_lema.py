import spacy
import random


def buscar_sentencas_por_lema(lema_alvo, lista_sentencas, N=None):
    """
    Busca, de forma aleatória, sentenças que contenham um lema-alvo específico.

    A lista de sentenças é embaralhada (sem alterar a original) e processada em lote
    com o spaCy. Para cada sentença, extrai-se o conjunto de lemas e verifica-se a
    presença exata do `lema_alvo`. A busca para assim que `N` ocorrências forem
    encontradas, evitando o processamento de todo o corpus.

    Observação:
        O `lema_alvo` é lematizado internamente antes da comparação, então pode ser
        informado em qualquer forma flexionada (ex.: "gatos" é reduzido a "gato";
        "correram" a "correr").

    Args:
        lema_alvo (str): Termo a ser procurado. É normalizado para minúsculas e
            lematizado internamente antes da busca.
        lista_sentencas (list of str): Corpus segmentado em sentenças.
        N (int, opcional): Número máximo de sentenças a retornar. Se None (padrão),
            varre todo o corpus e retorna todas as ocorrências encontradas.

    Returns:
        list of str: Lista (em ordem aleatória) das sentenças que contêm o lema-alvo,
        limitada a `N` itens. Retorna lista vazia se `lista_sentencas` for vazia ou
        se nenhuma ocorrência for encontrada.

    Dependências:
        Requer o spaCy e o modelo "pt_core_news_lg" instalados no ambiente.
    """
    if not lista_sentencas:
        return []

    # 1. Copia superficial para evitar a destruição da lista original e embaralha as posições
    copia_sentencas = lista_sentencas.copy()
    random.shuffle(copia_sentencas)

    # 2. Definição do limite
    limite = len(copia_sentencas) if N is None else N

    # 3. Preparação do modelo
    nlp = spacy.load("pt_core_news_lg", disable=["parser", "ner"])

    # Lematiza o próprio termo-alvo, permitindo passar formas flexionadas (ex.: "gatos" -> "gato")
    doc_alvo = nlp(lema_alvo.lower())
    lema_alvo = doc_alvo[0].lemma_.lower() if len(doc_alvo) > 0 else lema_alvo.lower()

    sentencas_encontradas = []

    # 4. Processamento em lote otimizado
    for doc in nlp.pipe(copia_sentencas, batch_size=50):
        lemas_frase = {token.lemma_.lower() for token in doc if token.is_alpha}

        if lema_alvo in lemas_frase:
            sentencas_encontradas.append(doc.text)

        # 5. Condição de paragem segura
        if len(sentencas_encontradas) >= limite:
            break

    return sentencas_encontradas