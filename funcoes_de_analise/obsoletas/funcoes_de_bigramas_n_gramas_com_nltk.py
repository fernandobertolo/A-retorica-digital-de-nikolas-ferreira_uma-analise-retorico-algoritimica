import csv
import datetime
import os
from nltk.probability import FreqDist
from nltk.collocations import BigramCollocationFinder
from nltk.tokenize import word_tokenize
from nltk.metrics import BigramAssocMeasures


def analisar_palavras_recorrentes_em_janela(
        palavra_alvo,
        lista_de_frases,
        janela=3,
        n=10,
        stopwords=None,
        metrica="loglikelihood"
):
    palavra_alvo = palavra_alvo.lower()
    sw = set(w.lower() for w in stopwords) if stopwords else set()

    # Instanciação com a classe correta exigida pela API do NLTK
    freq_total = FreqDist()
    bigram_fd = FreqDist()

    for frase in lista_de_frases:
        tokens_brutos = word_tokenize(frase.lower())
        tokens_limpos = [
            t for t in tokens_brutos
            if t.isalpha() and (t not in sw or t == palavra_alvo)
        ]

        if tokens_limpos:
            freq_total.update(tokens_limpos)

            if len(tokens_limpos) >= 2:
                temp_finder = BigramCollocationFinder.from_words(tokens_limpos, window_size=janela)
                bigram_fd.update(temp_finder.ngram_fd)

    freq_alvo = freq_total.get(palavra_alvo, 0)

    finder = BigramCollocationFinder(freq_total, bigram_fd, window_size=janela)

    finder.apply_ngram_filter(lambda w1, w2: palavra_alvo not in (w1, w2))

    metricas = {
        "loglikelihood": BigramAssocMeasures.likelihood_ratio,
        "pmi": BigramAssocMeasures.pmi,
        "chi_sq": BigramAssocMeasures.chi_sq
    }

    if metrica not in metricas:
        raise ValueError(f"Métrica inválida. Use: {list(metricas.keys())}")

    scores = finder.score_ngrams(metricas[metrica])

    agreg = {}
    for (w1, w2), score in scores:
        associada = w2 if w1 == palavra_alvo else w1

        freq_associada = freq_total.get(associada, 0)
        freq_coocorrencia = bigram_fd[(w1, w2)]

        atual = agreg.get(associada)
        if (atual is None) or (score > atual["score"]):
            agreg[associada] = {
                "score": round(score, 2),
                "freq_associada": freq_associada,
                "freq_coocorrencia": freq_coocorrencia
            }

    resultado = [
        (palavra, data["score"], data["freq_associada"], freq_alvo, data["freq_coocorrencia"])
        for palavra, data in agreg.items()
    ]
    resultado.sort(key=lambda x: x[1], reverse=True)

    print(f"{'Palavra Assoc.':<20} | {'Score':<10} | {'Freq Assoc.':<12} | {'Freq Alvo':<10} | {'Coocorrência':<12}")
    print("-" * 75)
    for palavra in resultado[:n]:
        print(f"{palavra[0]:<20} | {palavra[1]:<10.2f} | {palavra[2]:<12} | {palavra[3]:<10} | {palavra[4]:<12}")

    return resultado[:n]