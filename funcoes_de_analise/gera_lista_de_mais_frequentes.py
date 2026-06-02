import os
import math
from collections import Counter
import spacy


def gera_lista_de_mais_frequentes(lista_textos,
                                  caminho_corpus_referencia=None,
                                  n=50,
                                  classes_alvo={"NOUN", "ADJ", "VERB", "PROPN", "ADV", "INTJ"},
                                  limiar=0,
                                  lemma=True,
                                  stop_words={"a", "e", "i", "o", "u", "que", "de", "do", "da"}):
    """
    Processa uma lista de textos, extrai formas ativas e calibra as frequências
    contra um corpus de referência utilizando o cálculo de Log-Likelihood.

    Metodologia:
    1. Lê e processa tanto o corpus específico quanto o de referência usando spaCy (Tokenização Simétrica).
    2. Filtra tokens por classe gramatical (POS) e remove stop words na origem.
    3. Aplica o teste de Log-Likelihood para medir a "Keyness" (Especificidade) de cada termo.
    4. Exclui termos cujo índice de verossimilhança seja menor que o limiar estipulado.
    5. Recalcula a frequência relativa (%) com base apenas no vocabulário calibrado restante.

    Args:
        lista_textos (list of str): Lista contendo os textos do corpus específico a ser analisado.
        caminho_corpus_referencia (str): Caminho para um ficheiro .txt ou diretório do corpus geral.
        n (int): Número de termos mais frequentes a serem exibidos e retornados (Padrão: 50).
        classes_alvo (set): Conjunto de classes gramaticais (POS Tags) válidas para a análise.
        limiar (float): Nota de corte do Log-Likelihood. 0 = aceita tudo. (Padrão: 0).
        lemma (bool): Se True, lematiza os tokens. Se False, extrai a palavra original.
        stop_words (set/list): Lista de palavras a serem ignoradas na contagem.

    Returns:
        dict: Dicionário onde a chave é o termo (str) e o valor é a sua frequência relativa calibrada (float).
    """

    # 1. PREPARAÇÃO DO AMBIENTE
    nlp = spacy.load("pt_core_news_lg", disable=["parser", "ner"])
    stop_words_set = set(stop_words)

    # =====================================================================
    # FUNÇÕES INTERNAS
    # =====================================================================

    def carregar_textos_referencia(caminho):
        textos = []
        if caminho.endswith(".txt"):
            with open(caminho, "r", encoding="utf-8") as f:
                # Segmentação estrutural: iterador de linhas em vez de f.read()
                for linha in f:
                    if linha.strip():
                        textos.append(linha)
        else:
            for raiz, _, arquivos in os.walk(caminho):
                for arquivo in arquivos:
                    if arquivo.endswith(".txt"):
                        with open(os.path.join(raiz, arquivo), "r", encoding="utf-8") as f:
                            for linha in f:
                                if linha.strip():
                                    textos.append(linha)
        return textos

    def extrair_frequencias(textos):
        contador = Counter()
        for doc in nlp.pipe(textos, batch_size=50):
            for token in doc:
                if token.pos_ in classes_alvo and token.is_alpha:
                    termo = token.lemma_.lower() if lemma else token.text.lower()
                    if termo not in stop_words_set:
                        contador[termo] += 1
        return contador

    # 2. PROCESSAMENTO DOS CORPORA
    print("Processando corpus específico (aguarde)...")
    contador_especifico = extrair_frequencias(lista_textos)
    tamanho_especifico = sum(contador_especifico.values())

    contador_calibrado = Counter()

    if limiar != 0 and caminho_corpus_referencia:
        print("Processando corpus de referência (aguarde)...")
        textos_referencia = carregar_textos_referencia(caminho_corpus_referencia)
        contador_referencia = extrair_frequencias(textos_referencia)
        tamanho_referencia = sum(contador_referencia.values())

        # 3. CALIBRAÇÃO ESTATÍSTICA (LOG-LIKELIHOOD)
        for termo, oc in contador_especifico.items():
            og = contador_referencia.get(termo, 0)

            if og == 0:
                likelihood = float('inf')
            else:
                taxa_geral = og / tamanho_referencia
                oe = taxa_geral * tamanho_especifico

                try:
                    parte1 = oc * math.log(oc / oe) if oc > 0 else 0
                    diff_oc = tamanho_especifico - oc
                    diff_oe = tamanho_especifico - oe

                    parte2 = 0
                    if diff_oc > 0 and diff_oe > 0:
                        parte2 = diff_oc * math.log(diff_oc / diff_oe)

                    likelihood = 2 * (parte1 + parte2)
                except ValueError:
                    likelihood = 0

            if likelihood >= limiar:
                contador_calibrado[termo] = oc
    else:
        # Ignora o cálculo de verossimilhança e atribui a totalidade do corpus específico
        contador_calibrado = contador_especifico

    # 4. CÁLCULO DE MÉTRICAS FINAIS
    tamanho_calibrado = sum(contador_calibrado.values())
    mais_frequentes = contador_calibrado.most_common(n)

    dicionario_resultado = {}
    freq_acumulada = 0
    total_absoluto_exibido = 0

    # 5. RENDERIZAÇÃO DA TABELA (I/O)
    print(f"\n{'Nº':<4} {'Termo':<17} {'Absoluta':<12} {'Relativa (%)':<12}")
    print("-" * 50)

    for i, (termo, contagem) in enumerate(mais_frequentes, 1):
        if tamanho_calibrado > 0:
            freq_relativa = round((contagem / tamanho_calibrado) * 100, 2)
        else:
            freq_relativa = 0.0

        dicionario_resultado[termo] = freq_relativa
        freq_acumulada += freq_relativa
        total_absoluto_exibido += contagem

        print(f"{i:<4} {termo:<17} {contagem:<12} {freq_relativa:<12}")

    print("-" * 50)
    print(f"{'Total (Amostra)':<22} {total_absoluto_exibido:<12} {round(freq_acumulada, 2):<12}")
    print(f"\nVolumetria do Corpus Específico:")
    print(f" - Tokens válidos originais: {tamanho_especifico}")
    print(f" - Tokens após calibração  : {tamanho_calibrado}")

    return dicionario_resultado