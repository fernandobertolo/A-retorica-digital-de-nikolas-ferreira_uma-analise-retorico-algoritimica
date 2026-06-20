from collections import Counter
from itertools import combinations


def ngramas_spacy(alvos, lista_de_sentencas, nlp_model, classes_alvo={"NOUN", "ADJ", "VERB", "PROPN", "ADV", "INTJ"},
                  N=3, janela=3, lemma=False):
    """
    Identifica coocorrências e gera n-gramas relacionados a termos-alvo utilizando processamento em lote do spaCy e extração KWIC.

    A função aplica uma extração de contexto via Key Word In Context (KWIC) sobre as sentenças fornecidas para extrair
    os termos vizinhos aos alvos, filtrando por classes gramaticais (POS tags) e
    contabilizando suas frequências absolutas. O processamento é estruturado em lote utilizando `nlp.pipe`.

    Args:
        alvos (list of str): Lista de termos-alvo para a busca de associações no texto.
        lista_de_sentencas (list of str): Corpus de entrada segmentado em sentenças.
        nlp_model (spacy.Language): Modelo de linguagem spaCy instanciado para processamento otimizado.
        classes_alvo (set, opcional): Conjunto de classes morfológicas válidas para a
            composição dos n-gramas. Padrão: {"NOUN", "ADJ", "VERB", "PROPN", "ADV", "INTJ"}.
        N (int, opcional): Número de palavras que compõem o n-grama final. Padrão: 3.
        janela (int, opcional): Amplitude da janela de contexto (palavras à esquerda e à direita da palavra-alvo). Padrão: 3.
        lemma (bool, opcional): Determina se os tokens do corpus e os termos declarados em `alvos` serão
            reduzidos aos seus lemas antes da correspondência e contagem. Padrão: False.

    Returns:
        list of tuple: Estrutura de dados contendo as associações e frequências, apropriada para serialização
        ou conversão estruturada (ex: DataFrame).
        Formato: [((alvo, vizinho_1, vizinho_2), frequencia), ...]

    Fluxo de Execução:
        1. Normalização morfológica dos termos em `alvos` (condicional à flag `lemma`).
        2. Ingestão e tokenização do corpus em lote via `nlp_model.pipe`.
        3. Localização do termo-alvo e extração exata de vizinhos via estrutura de janela (KWIC nativo no Doc).
        4. Filtragem de tokens que não pertencem ao conjunto `classes_alvo`.
        5. Agregação estatística.
        6. Impressão estruturada do relatório e retorno dos dados ordenados.
    """

    alvos_processados = set()
    if lemma:
        for alvo in alvos:
            doc_alvo = nlp_model(alvo.lower())
            if len(doc_alvo) > 0:
                alvos_processados.add(doc_alvo[0].lemma_)
    else:
        alvos_processados = {alvo.lower() for alvo in alvos}

    frequencias = Counter()
    documentos = nlp_model.pipe(lista_de_sentencas, batch_size=100)

    for doc in documentos:
        tamanho_doc = len(doc)

        for i, token in enumerate(doc):
            termo_atual = token.lemma_.lower() if lemma else token.text.lower()

            if termo_atual in alvos_processados:
                inicio_janela = max(0, i - janela)
                fim_janela = min(tamanho_doc, i + janela + 1)

                vizinhos_validos = []
                for vizinho in doc[inicio_janela:fim_janela]:
                    if vizinho.i == i:
                        continue

                    if vizinho.pos_ in classes_alvo and vizinho.is_alpha:
                        termo_vizinho = vizinho.lemma_.lower() if lemma else vizinho.text.lower()
                        vizinhos_validos.append(termo_vizinho)

                if len(vizinhos_validos) >= (N - 1):
                    for combinacao in combinations(vizinhos_validos, N - 1):
                        ngrama = (termo_atual,) + combinacao
                        frequencias[ngrama] += 1

    resultados = frequencias.most_common()

    print(f"{'Alvos':<20} | {'N-gramas':<40} | {'Ocorrências':<15}")
    print("-" * 85)
    for ngrama, freq in resultados:
        alvo_str = f"[{ngrama[0]}]"
        vizinhos_str = " ".join(ngrama[1:])
        print(f"{alvo_str:<20} | {vizinhos_str:<40} | {freq:<15}")

    return resultados