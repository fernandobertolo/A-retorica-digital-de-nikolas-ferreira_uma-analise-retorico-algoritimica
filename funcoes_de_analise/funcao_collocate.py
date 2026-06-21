import math
import spacy
from collections import Counter, defaultdict

# Requisito de execução:
# É necessário ter o spaCy e o modelo em português instalados no ambiente:
# pip install spacy
# python -m spacy download pt_core_news_sm

try:
    nlp = spacy.load("pt_core_news_sm")
except OSError:
    raise OSError("Modelo linguístico não encontrado. Execute: python -m spacy download pt_core_news_sm")


def extrair_collocates_antconc(
        termos_alvo: list,
        sentencas: list,
        classes_morfosintaticas: set,
        janela_esquerda: int,
        janela_direita: int,
        ordenar_por: str = "Likelihood",
        lematizar: bool = False
):
    """
    Emula a ferramenta Collocates do AntConc.

    Argumentos:
    1 - termos_alvo: Lista de strings representando os nódulos.
    2 - sentencas: Lista de strings (corpus fragmentado).
    3 - classes_morfosintaticas: Set contendo as tags POS permitidas.
        Tags suportadas (Universal POS tags do spaCy):
        'ADJ'   (adjetivo)
        'ADP'   (adposição / preposição)
        'ADV'   (advérbio)
        'AUX'   (verbo auxiliar)
        'CCONJ' (conjunção coordenativa)
        'DET'   (determinante / artigo)
        'INTJ'  (interjeição)
        'NOUN'  (substantivo)
        'NUM'   (numeral)
        'PART'  (partícula)
        'PRON'  (pronome)
        'PROPN' (nome próprio)
        'PUNCT' (pontuação)
        'SCONJ' (conjunção subordinativa)
        'SYM'   (símbolo)
        'VERB'  (verbo)
        'X'     (outro)
        'SPACE' (espaço)
    4 - janela_esquerda: Inteiro definindo o limite de tokens à esquerda.
    5 - janela_direita: Inteiro definindo o limite de tokens à direita.
    6 - ordenar_por: String indicando a coluna de ordenação.
        Opções suportadas: 'Rank', 'FreqLR', 'FreqL', 'FreqR', 'Range', 'Likelihood', 'Effect'.
    7 - lematizar: Booleano indicando se os tokens devem ser lematizados (True) ou mantidos na forma superficial (False).
    """

    nodulos = set(n.lower() for n in termos_alvo)

    # Contadores globais
    freq_global_corpus = Counter()
    freq_alvos = Counter()

    # Contadores de coocorrência. Chave estrutural: (nódulo, colocado)
    freq_L = Counter()
    freq_R = Counter()
    range_docs = defaultdict(set)

    tamanho_total_corpus = 0
    total_sentencas = len(sentencas)

    # 1. Varredura do corpus e indexação otimizada com nlp.pipe()
    # Desativa os componentes parser e ner para aumentar a velocidade de processamento
    for idx_sentenca, doc in enumerate(nlp.pipe(sentencas, disable=["parser", "ner"])):
        # Marcador de progresso
        if total_sentencas > 0 and (idx_sentenca + 1) % max(1, total_sentencas // 100) == 0:
            progresso = ((idx_sentenca + 1) / total_sentencas) * 100
            print(f"\rProcessando corpus: {progresso:.1f}% concluído [{idx_sentenca + 1}/{total_sentencas}]", end="",
                  flush=True)

        tokens_processados = []

        # Tokenização e extração morfosintática
        for token in doc:
            if not token.is_space and not token.is_punct:
                # Aplicação condicional da lematização
                texto_token = token.lemma_.lower() if lematizar else token.text.lower()
                pos_token = token.pos_
                tokens_processados.append((texto_token, pos_token))

                # O corpus total (N) contabiliza apenas as classes válidas e os próprios alvos
                if pos_token in classes_morfosintaticas or texto_token in nodulos:
                    freq_global_corpus[texto_token] += 1
                    tamanho_total_corpus += 1

        # 2. Mapeamento das janelas e extração de frequências relativas
        for i, (texto, pos) in enumerate(tokens_processados):
            if texto in nodulos:
                freq_alvos[texto] += 1

                # Processamento da Janela Esquerda
                limite_inicio_esq = max(0, i - janela_esquerda)
                for j in range(limite_inicio_esq, i):
                    t_esq, pos_esq = tokens_processados[j]
                    if pos_esq in classes_morfosintaticas:
                        freq_L[(texto, t_esq)] += 1
                        range_docs[(texto, t_esq)].add(idx_sentenca)

                # Processamento da Janela Direita
                limite_fim_dir = min(len(tokens_processados), i + janela_direita + 1)
                for j in range(i + 1, limite_fim_dir):
                    t_dir, pos_dir = tokens_processados[j]
                    if pos_dir in classes_morfosintaticas:
                        freq_R[(texto, t_dir)] += 1
                        range_docs[(texto, t_dir)].add(idx_sentenca)

    print("\nCalculando métricas estatísticas...")

    # 3. Cálculo das métricas estatísticas
    resultados = []
    pares_registrados = set(freq_L.keys()).union(set(freq_R.keys()))

    for (nodulo, colocado) in pares_registrados:
        f_L = freq_L.get((nodulo, colocado), 0)
        f_R = freq_R.get((nodulo, colocado), 0)
        f_LR = f_L + f_R
        f_range = len(range_docs[(nodulo, colocado)])

        f_node = freq_alvos[nodulo]
        f_col = freq_global_corpus[colocado]

        # Cálculo de Mutual Information (Effect Size)
        effect = 0.0
        if f_node > 0 and f_col > 0 and f_LR > 0:
            effect = math.log2((f_LR * tamanho_total_corpus) / (f_node * f_col))

        # Cálculo do Log-Likelihood Ratio (Dunning G2)
        a = f_LR
        b = max(f_node - a, 0)
        c = max(f_col - a, 0)
        d = max(tamanho_total_corpus - a - b - c, 0)

        E_a = ((a + b) * (a + c)) / tamanho_total_corpus if tamanho_total_corpus > 0 else 0
        E_b = ((a + b) * (b + d)) / tamanho_total_corpus if tamanho_total_corpus > 0 else 0
        E_c = ((c + d) * (a + c)) / tamanho_total_corpus if tamanho_total_corpus > 0 else 0
        E_d = ((c + d) * (b + d)) / tamanho_total_corpus if tamanho_total_corpus > 0 else 0

        def iteracao_log(obs, exp):
            return obs * math.log(obs / exp) if obs > 0 and exp > 0 else 0.0

        ll = 2 * (iteracao_log(a, E_a) + iteracao_log(b, E_b) + iteracao_log(c, E_c) + iteracao_log(d, E_d))

        resultados.append({
            "Nodulo": nodulo,
            "Colocado": colocado,
            "FreqLR": f_LR,
            "FreqL": f_L,
            "FreqR": f_R,
            "Range": f_range,
            "Likelihood": round(ll, 2),
            "Effect": round(effect, 2)
        })

    # 4. Ordenação algorítmica
    chaves_validas = ["Rank", "FreqLR", "FreqL", "FreqR", "Range", "Likelihood", "Effect"]

    if ordenar_por not in chaves_validas:
        raise ValueError(f"Parâmetro de ordenação inválido. Use um dos seguintes: {chaves_validas}")

    # Se 'Rank' for selecionado, a ordenação padrão é por 'Likelihood' decrescente
    chave_ordenacao = "Likelihood" if ordenar_por == "Rank" else ordenar_por

    resultados.sort(key=lambda x: x[chave_ordenacao], reverse=True)

    # Inserção do índice de Rank e exibição na saída padrão
    print(
        f"\n{'Rank':<5} | {'Nódulo':<15} | {'Colocado':<15} | {'FreqLR':<8} | {'FreqL':<7} | {'FreqR':<7} | {'Range':<6} | {'Likelihood':<10} | {'Effect':<8}")
    print("-" * 95)

    for indice, res in enumerate(resultados):
        res["Rank"] = indice + 1
        print(
            f"{res['Rank']:<5} | {res['Nodulo']:<15} | {res['Colocado']:<15} | {res['FreqLR']:<8} | {res['FreqL']:<7} | {res['FreqR']:<7} | {res['Range']:<6} | {res['Likelihood']:<10.2f} | {res['Effect']:<8.2f}")

    return resultados

# Exemplo de execução estruturada com lematização ativada:
# sentencas_teste = ["O estado forte garante a liberdade do povo.", "Sem o estado, a liberdade perece."]
# classes_alvo = {"NOUN", "ADJ", "VERB"}
# alvo = ["liberdade"]
# extrair_collocates_antconc(alvo, sentencas_teste, classes_alvo, 3, 3, "Likelihood", lematizar=True)