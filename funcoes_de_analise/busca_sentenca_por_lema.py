import spacy
import random


def buscar_sentencas_por_lema(lema_alvo, lista_sentencas, N=None):
    """
    Processa as sentenças de forma aleatória e verifica a intersecção exata de lemas,
    retornando um limite de N ocorrências.
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
    lema_alvo = lema_alvo.lower()

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