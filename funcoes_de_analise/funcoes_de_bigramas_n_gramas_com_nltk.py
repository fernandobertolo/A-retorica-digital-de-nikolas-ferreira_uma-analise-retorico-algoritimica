from funcoes_de_analise.stop_words import stop_words
import random, csv, datetime
from nltk.collocations import BigramCollocationFinder
from nltk.tokenize import word_tokenize
from nltk.metrics import BigramAssocMeasures
from collections import Counter



def analisar_palavras_recorrentes_em_janela(
    palavra_alvo,
    lista_de_frases,
    janela=3,
    n=10,
    stopwords=None,
    metrica="loglikelihood"
):
    """
        Analisa palavras recorrentes no contexto de uma palavra-alvo usando bigramas
        em uma janela deslizante, calculando a força de associação estatística e
        incluindo informações de frequência.

        Esta função permite identificar quais palavras aparecem mais frequentemente
        próximas de uma palavra de interesse em um corpus de texto. O cálculo da
        associação pode ser feito usando diferentes métricas estatísticas (log-likelihood, PMI ou qui-quadrado).
        Ela também retorna a frequência de cada palavra no corpus e a frequência com que
        cada palavra aparece junto da palavra-alvo.

        Passos principais:
        1. Tokeniza todas as frases e normaliza (tudo em minúsculas, remove pontuação e números).
        2. Remove stopwords opcionais, mas mantém a palavra-alvo.
        3. Calcula a frequência total da palavra-alvo e de todas as outras palavras.
        4. Cria bigramas dentro de uma janela deslizante definida pelo usuário.
        5. Filtra apenas os bigramas que contêm a palavra-alvo.
        6. Calcula a métrica de associação escolhida para cada bigrama.
        7. Agrega resultados por palavra associada (mantendo o maior score caso haja múltiplos bigramas).
        8. Ordena os resultados pelo score de associação e retorna os top-n resultados.

        Parâmetros:
        -----------
        palavra_alvo : str
            Palavra de interesse para análise de coocorrência.
        lista_de_frases : list of str
            Corpus textual, onde cada elemento da lista é uma frase.
        janela : int, opcional (default=3)
            Número de palavras antes e depois da palavra-alvo que serão consideradas para formar bigramas.
        n : int, opcional (default=10)
            Número máximo de palavras associadas a serem retornadas.
        stopwords : list of str, opcional (default=None)
            Lista de palavras a serem ignoradas na análise, exceto a palavra-alvo.
        metrica : str, opcional (default="loglikelihood")
            Métrica estatística usada para medir a força de associação entre palavras.
            Opções válidas:
                - "loglikelihood" : Log-Likelihood Ratio, robusto para grandes corpora.
                - "pmi"           : Pointwise Mutual Information, destaca associações raras.
                - "chi_sq"        : Qui-quadrado, mede independência entre palavras.

        Retorno:
        --------
        list of tuples
            Cada tupla contém cinco elementos:
            [
                (palavra_associada, score, freq_associada, freq_alvo, freq_coocorrencia), ...
            ]

            Onde:
            - palavra_associada : str
                Palavra que aparece junto da palavra-alvo dentro da janela definida.
            - score : float
                Valor da métrica de associação escolhida para o bigrama (quanto maior, mais forte a associação).
            - freq_associada : int
                Frequência total da palavra associada no corpus.
            - freq_alvo : int
                Frequência total da palavra-alvo no corpus.
            - freq_coocorrencia : int
                Número de vezes que a palavra-alvo aparece junto da palavra associada (coocorrência no bigrama).

        Observações:
        ------------
        - A função imprime os top-n resultados em formato de tabela organizada.
        - É possível usar uma lista de stopwords para remover palavras comuns que não interessam na análise.
        - A métrica de associação pode ser ajustada conforme o tipo de corpus e objetivo da análise.
    """


    # 1) Tokenização e normalização
    tokens = []
    for frase in lista_de_frases:
        tokens.extend(word_tokenize(frase.lower()))

    tokens = [t for t in tokens if t.isalpha()]  # Remove pontuação/números

    # 2) Remove stopwords, preservando a palavra_alvo
    if stopwords:
        sw = set(w.lower() for w in stopwords)
        tokens = [t for t in tokens if (t not in sw) or (t == palavra_alvo)]

    # 3) Frequências globais
    freq_total = Counter(tokens)
    freq_alvo = freq_total.get(palavra_alvo, 0)

    # 4) Finder de bigramas com janela
    finder = BigramCollocationFinder.from_words(tokens, window_size=janela)

    # Filtra para manter apenas bigramas com a palavra_alvo
    finder.apply_ngram_filter(lambda w1, w2: palavra_alvo not in (w1, w2))

    # 5) Métricas
    metricas = {
        "loglikelihood": BigramAssocMeasures.likelihood_ratio,
        "pmi": BigramAssocMeasures.pmi,
        "chi_sq": BigramAssocMeasures.chi_sq
    }
    if metrica not in metricas:
        raise ValueError(f"Métrica inválida. Use: {list(metricas.keys())}")
    score_fn = metricas[metrica]

    # 6) Calcula scores
    scores = finder.score_ngrams(score_fn)

    # 7) Frequência de bigramas
    freq_bigrams = finder.ngram_fd  # Counter com contagem dos bigramas

    # 8) Agrega por palavra associada
    agreg = {}
    for (w1, w2), score in scores:
        if w1 == palavra_alvo and w2 != palavra_alvo:
            associada = w2
        elif w2 == palavra_alvo and w1 != palavra_alvo:
            associada = w1
        else:
            continue

        freq_associada = freq_total.get(associada, 0)
        freq_coocorrencia = freq_bigrams[(w1, w2)]

        atual = agreg.get(associada)
        if (atual is None) or (score > atual["score"]):
            agreg[associada] = {
                "score": score,
                "freq_associada": freq_associada,
                "freq_coocorrencia": freq_coocorrencia
            }

    # 9) Ordena e retorna
    resultado = [
        (palavra, data["score"], data["freq_associada"], freq_alvo, data["freq_coocorrencia"])
        for palavra, data in agreg.items()
    ]
    resultado.sort(key=lambda x: x[1], reverse=True)

    print(f"{'palavra assoc.':<20}{'score':<20}{'freq_assoc.':<10}{'freq_alvo':<10}{'freq_coocorrencia':10}")


    for palavra in resultado[:n]:
        print(f'{palavra[0]:<20}{palavra[1]:<20}{palavra[2]:<10}{palavra[3]:<10}{palavra[4]:<10}')

    #return resultado[:n]


def gera_bigramas(mensagens_selecionadas, janela, palavra_alvo=None):
    # 1 - Recebe a lista de comentários
    lista_de_mensagens = mensagens_selecionadas

    # 2 - Seleciona apenas as mensagens com a palavra alvo
    mensagens_alvo = []
    if palavra_alvo:
        for mensagem in lista_de_mensagens:
            if palavra_alvo.lower() in mensagem.lower():
                mensagens_alvo.append(mensagem)
    else:
        mensagens_alvo = lista_de_mensagens

    print(f" Foram selecionado {len(mensagens_alvo)} em mensagens_alvo")

    # 3 - Bigramas
    stopwords = stop_words()

    # Criar um novo finder vazio
    all_finder = BigramCollocationFinder.from_words([])

    for mensagem in mensagens_alvo:
        # Toqueniza o comentário
        tokens = word_tokenize(mensagem.lower())
        # Filtra os tokens que são stopwords
        tokens_filtrados = [t for t in tokens if t not in stopwords]
        # Conta os brigramas do comentário
        finder = BigramCollocationFinder.from_words(tokens_filtrados, window_size=janela)

        # Adicionar frequências do finder atual ao all_finder
        for bigrama, freq in finder.ngram_fd.items():
            # O objeto BigramCollocarionFinder aceita soma, por isso o código incrementa diretamente a virável do objto
            all_finder.ngram_fd[bigrama] += freq

    print(f"{'N-grama':40} Frequência")
    print("-"*55)
    for ngrama, freq in all_finder.ngram_fd.most_common(16):
        frase = " ".join(ngrama)
        print(f"{frase:<40} {freq}")


    ### Salvar em CSV
    # Pega a data atual
    data_atual = datetime.date.today()
    # Formata no padrão 'aaaammdd'
    data_formatada = data_atual.strftime("%Y%m%d")
    nome_arquivo = f"saidas\\bigrams_frequencia_de_msg_{data_atual}.csv"
    with open(nome_arquivo, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['N-grama', 'Frequência'])
        for ngrama, freq in all_finder.ngram_fd.most_common(50):
            frase = " ".join(ngrama)
            writer.writerow([frase, freq])

    print(f"\nOs dados foram salvos em {nome_arquivo}")