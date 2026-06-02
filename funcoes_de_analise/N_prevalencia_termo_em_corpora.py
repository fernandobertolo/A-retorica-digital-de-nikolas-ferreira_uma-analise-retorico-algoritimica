import spacy

nlp = spacy.load('pt_core_news_sm')


# 1 - Entra uma lista de termos
# 2 - Entra uma lista de corpora
# 3 - lemmatiza os termos
# 4 - Lemmatiza os corpora
# 5 - Itera os termos
# 6 - Verifica se o termo está presente no corpora
# 7 - Adiciona um marcador em um dicionário com chave(termo) e valor (total , %)
# 8 - Retorna o dicionário {termo:(433, 78%)}

def prevalencia_termo_em_corpora(lista_termos, lista_corpora):
    """
    Calcula a prevalência de termos lematizados em uma lista de textos (corpora).
    """

    # 3 - Lematiza os termos (removendo duplicatas para não contar o mesmo termo duas vezes)
    doc_termos = nlp(" ".join(lista_termos))
    lemas_termos = {token.lemma_.lower() for token in doc_termos if token.is_alpha}

    # 4 - Lematiza os corpora
    # Armazenamos como lista de SETS para busca O(1) - muito mais rápido que lista
    lista_sets_corpora = []
    for i, corpus in enumerate(lista_corpora):
        print(f"Iniciando corpus {i+1}/{len(lista_corpora)}")
        doc = nlp(corpus)
        lemas_corpus = {token.lemma_.lower() for token in doc if token.is_alpha}
        lista_sets_corpora.append(lemas_corpus)

    num_total_corpora = len(lista_corpora)
    dicionario_final = {}

    # 5, 6, 7 e 8 - Itera termos e verifica presença
    for i, termo in enumerate(lemas_termos):
        print(f"Iniciando termo {i+1}/{num_total_corpora}")
        placar_termo = sum(1 for set_corpus in lista_sets_corpora if termo in set_corpus)

        # Calcula porcentagem
        porcentagem = (placar_termo / num_total_corpora) * 100

        # Formata o valor como tupla (total, texto_formatado) ou (total, float)
        # Sugestão: manter o float para cálculos futuros e formatar apenas na exibição
        dicionario_final[termo] = (placar_termo, f"{porcentagem:.1f}%")

    return dicionario_final