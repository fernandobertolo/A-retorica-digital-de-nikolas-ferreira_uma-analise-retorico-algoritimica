import spacy
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

# Carregamento do modelo linguístico (com parser + vetores, necessarios para
# a analise de dependencias e para o HDBSCAN)
try:
    nlp = spacy.load("pt_core_news_lg")
except OSError:
    raise OSError("Modelo linguístico não encontrado. Execute: python -m spacy download pt_core_news_lg")


def _chave_alvo(texto, lemma):
    """Chave de comparacao de um alvo: seu lema (se lemma=True) ou a forma minuscula."""
    if lemma:
        doc = nlp(texto)
        for token in doc:
            if token.is_alpha:
                return token.lemma_.lower()
        return texto.lower()
    return texto.lower()


def _sentenca_representativa_hdbscan(docs):
    """
    Etapa 6: dado um conjunto de docs spaCy, agrupa por similaridade semantica
    (vetores de contexto) com HDBSCAN e retorna a sentenca medoide do maior
    cluster (a mais proxima do centroide). Faz fallback seguro para poucos dados.
    """
    if not docs:
        return ""
    textos = [d.text for d in docs]
    if len(docs) == 1:
        return textos[0]

    vetores = np.array([d.vector for d in docs], dtype=float)

    idxs = list(range(len(docs)))  # fallback padrao: considera todos
    try:
        import hdbscan
        clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric="euclidean")
        labels = clusterer.fit_predict(vetores)
        rotulos_validos = [l for l in labels if l >= 0]
        if rotulos_validos:
            # maior cluster, desconsiderando o ruido (-1)
            maior = Counter(rotulos_validos).most_common(1)[0][0]
            idxs = [i for i, l in enumerate(labels) if l == maior]
    except Exception:
        pass

    sub = vetores[idxs]
    centroide = sub.mean(axis=0)
    distancias = np.linalg.norm(sub - centroide, axis=1)
    melhor = idxs[int(distancias.argmin())]
    return textos[melhor]


def analise_completa_por_tags(sentencas, alvos, nome_do_arquivo="analise.csv", limiar=0.3, lemma=True):
    """
    Análise sintática completa dos termos-alvo em um corpus, com exportação para CSV.

    Etapas:
      2. Filtra as sentenças que contêm os alvos (por lema se lemma=True).
      3. Conta as POS tags (.pos_) que o alvo exerce.
      4. Mapeia dinamicamente a árvore de dependências: para cada ocorrência do alvo,
         coleta o nó pai (.head) e os filhos diretos (.children), consolidando a
         contagem de cada combinação POS|dep. Ordena por frequência e retém o grupo
         do topo cuja frequência acumulada alcança o `limiar`.
      5. Para cada combinação retida, amostra os termos vizinhos mais frequentes (lematizados).
      6. Seleciona a sentença representativa de cada combinação via HDBSCAN (medoide do
         maior cluster) sobre os vetores das sentenças que contêm o alvo + a combinação.
      7. Exibe uma tabela e salva o CSV.

    Args:
        sentencas (list of str): Corpus segmentado em sentenças.
        alvos (set of str): Termos a pesquisar.
        nome_do_arquivo (str): Nome do arquivo CSV de saída. Padrão: "analise.csv".
        limiar (float): Ponto de corte por frequência acumulada, entre 0 e 1. Ordena as
            combinações da mais para a menos frequente e retém o grupo do topo cuja soma
            de ocorrências alcança/ultrapassa esse percentual. Padrão: 0.3.
        lemma (bool): Se True, a busca e as análises consideram as formas lematizadas.

    Returns:
        pandas.DataFrame: Tabela com as colunas:
            - alvo
            - tag_alvo: POS dominante do alvo
            - pct_tag_alvo: % que a POS dominante representa do total de ocorrências do alvo
            - tag_dependente_prevalente: combinação POS | dep vizinha retida pelo limiar
            - pct_tag_dependente: % que essa combinação representa do total de dependentes
            - amostra_termos: termos vizinhos mais frequentes (forma bruta, não lematizada)
            - pct_amostra: % que a amostra exibida representa do total de ocorrências da combinação
            - sentenca_media_hdbscan: sentença representativa (medoide via HDBSCAN)
    """
    def _pct(fracao):
        return f"{fracao * 100:.1f}%"

    alvos = list(alvos)
    chaves_alvo = {alvo: _chave_alvo(alvo, lemma) for alvo in alvos}
    conjunto_chaves = set(chaves_alvo.values())

    def token_bate(token):
        forma = token.lemma_.lower() if lemma else token.text.lower()
        return forma if forma in conjunto_chaves else None

    # ------------------------------------------------------------------
    # Etapa 2: filtra e guarda os docs que contêm algum alvo
    # ------------------------------------------------------------------
    docs_filtrados = []
    total = len(sentencas)
    for i, doc in enumerate(nlp.pipe(sentencas, batch_size=50)):
        if any(token_bate(t) for t in doc):
            docs_filtrados.append(doc)
        # Etapa 0: progresso com retorno de carro
        if total > 0 and (i + 1) % max(1, total // 100) == 0:
            pct = ((i + 1) / total) * 100
            print(f"\rProcessando sentenças: {pct:.1f}% concluído [{i + 1}/{total}]", end="", flush=True)
    if total > 0:
        print()

    linhas = []
    N_AMOSTRA = 10  # nº de termos vizinhos amostrados por combinação

    # ------------------------------------------------------------------
    # Analise por alvo
    # ------------------------------------------------------------------
    for alvo in alvos:
        chave = chaves_alvo[alvo]

        contagem_pos_alvo = Counter()                 # Etapa 3
        contagem_combo = Counter()                    # Etapa 4: (POS|dep) vizinhos
        termos_por_combo = defaultdict(Counter)       # Etapa 5: lemas vizinhos por combo
        docs_por_combo = defaultdict(list)            # Etapa 6: sentenças por combo

        for doc in docs_filtrados:
            for token in doc:
                forma = token.lemma_.lower() if lemma else token.text.lower()
                if forma != chave:
                    continue

                # Etapa 3: POS do alvo
                contagem_pos_alvo[token.pos_] += 1

                # Etapa 4: nós conectados (pai + filhos diretos), exceto o próprio token
                vizinhos = []
                if token.head.i != token.i:
                    vizinhos.append(token.head)
                vizinhos.extend(token.children)

                for viz in vizinhos:
                    combo = f"{viz.pos_} | {viz.dep_}"
                    contagem_combo[combo] += 1
                    termos_por_combo[combo][viz.text.lower()] += 1  # termo bruto (sem lematizar)
                    docs_por_combo[combo].append(doc)

        if not contagem_combo:
            continue  # alvo não encontrado / sem vizinhos

        tag_alvo = contagem_pos_alvo.most_common(1)[0][0]
        # % que a POS dominante representa do total de ocorrências do alvo
        pct_tag_alvo = _pct(contagem_pos_alvo[tag_alvo] / sum(contagem_pos_alvo.values()))

        # ---- Etapa 4 (corte): frequência acumulada até o limiar ----
        total_combo = sum(contagem_combo.values())
        combos_ordenados = contagem_combo.most_common()
        retidas = []
        acumulado = 0
        for combo, cnt in combos_ordenados:
            retidas.append(combo)
            acumulado += cnt
            if acumulado / total_combo >= limiar:
                break

        # ---- Etapas 5 e 6 por combinação retida ----
        for combo in retidas:
            # % que esta combinação representa em relação a todos os dependentes (a amostra)
            pct_tag_dependente = _pct(contagem_combo[combo] / total_combo)

            # Etapa 5: termos vizinhos mais frequentes (termo bruto, sem lematizar)
            top = termos_por_combo[combo].most_common(N_AMOSTRA)
            amostra = [termo for termo, _ in top]
            # % que a amostra (top-N termos exibidos) representa do total de ocorrências da combinação
            total_termos_combo = sum(termos_por_combo[combo].values())
            pct_amostra = _pct(sum(c for _, c in top) / total_termos_combo)

            # Etapa 6: sentença representativa via HDBSCAN (docs únicos)
            docs_unicos = list({id(d): d for d in docs_por_combo[combo]}.values())
            sentenca_media = _sentenca_representativa_hdbscan(docs_unicos)

            linhas.append({
                "alvo": alvo,
                "tag_alvo": tag_alvo,
                "pct_tag_alvo": pct_tag_alvo,
                "tag_dependente_prevalente": combo,
                "pct_tag_dependente": pct_tag_dependente,
                "amostra_termos": amostra,
                "pct_amostra": pct_amostra,
                "sentenca_media_hdbscan": sentenca_media,
            })

    # ------------------------------------------------------------------
    # Etapa 7: tabela + CSV
    # ------------------------------------------------------------------
    df = pd.DataFrame(linhas, columns=[
        "alvo", "tag_alvo", "pct_tag_alvo",
        "tag_dependente_prevalente", "pct_tag_dependente",
        "amostra_termos", "pct_amostra",
        "sentenca_media_hdbscan",
    ])

    print(df.to_string(index=False))
    df.to_csv(nome_do_arquivo, index=False, encoding="utf-8-sig")
    print(f"\nArquivo salvo: {nome_do_arquivo}  ({len(df)} linhas)")

    return df
