import spacy
import matplotlib.pyplot as plt
import numpy as np

# Carregamento do modelo linguístico
try:
    nlp = spacy.load("pt_core_news_sm")
except OSError:
    raise OSError("Modelo linguístico não encontrado. Execute: python -m spacy download pt_core_news_sm")


def grafico_distribuicao_de_um_termo(lista_termos, lista_de_textos, lemma=False):
    """
    Esta função plota um gráfico de linhas que mostra em que parte de diversos textos (começo, meio, fim)
    cada termo de uma lista aparece mais. Dessa forma é possível saber se um termo específico está mais atrelado
    à introdução, ao desenvolvimento ou à conclusão.

    1 - A função itera a lista de textos transformando cada uma em uma lista de strings.
    2 - Itera as listas de string buscando os alvos, que podem ou não estar lematizados.
    3 - Caso encontre o termo, a função pega o índice e anota a porcentagem referente em uma lista específica para aquele termo.
    4 - A função plota um gráfico com múltiplas linhas, onde cada linha representa a distribuição espacial de um termo diferente.

    :argumentos:
    lista_termos(list): Lista de strings com os termos alvos.
    lista_de_textos(list): lista de strings com os textos alvos.
    lemma(bool): Determina se a busca deve lematizar tanto os termos quanto os textos.

    :return:
    Plot do gráfico
    """

    # 1. Processamento e padronização dos alvos
    alvos_processados = set()
    if lemma:
        for doc in nlp.pipe(lista_termos, disable=["parser", "ner"]):
            for token in doc:
                alvos_processados.add(token.lemma_.lower())
    else:
        alvos_processados = set(termo.lower() for termo in lista_termos)

    # Dicionário para armazenar as posições percentuais agrupadas por termo
    posicoes_por_termo = {alvo: [] for alvo in alvos_processados}
    total_textos = len(lista_de_textos)

    # 2. Varredura dos textos e busca dos alvos (nlp.pipe utilizado para performance)
    for idx, doc in enumerate(nlp.pipe(lista_de_textos, disable=["parser", "ner"])):
        # Marcador de progresso
        if total_textos > 0 and (idx + 1) % max(1, total_textos // 100) == 0:
            progresso = ((idx + 1) / total_textos) * 100
            print(f"\rProcessando textos: {progresso:.1f}% concluído [{idx + 1}/{total_textos}]", end="", flush=True)

        # Filtra espaços e pontuações para calcular o tamanho real de palavras do texto
        tokens_validos = [t for t in doc if not t.is_space and not t.is_punct]
        total_tokens = len(tokens_validos)

        if total_tokens == 0:
            continue

        # 3. Cálculo da porcentagem para os alvos encontrados
        for i, token in enumerate(tokens_validos):
            forma_analisada = token.lemma_.lower() if lemma else token.text.lower()

            if forma_analisada in posicoes_por_termo:
                # O índice zero representa 0%, e o último token aproxima-se de 100%
                porcentagem = (i / max(1, (total_tokens - 1))) * 100
                posicoes_por_termo[forma_analisada].append(porcentagem)

    print("\nConcluído. Processando gráfico...")

    # Verifica se houve detecção de termos globalmente
    houve_deteccao = any(len(posicoes) > 0 for posicoes in posicoes_por_termo.values())
    if not houve_deteccao:
        print("Nenhum dos termos alvos foi localizado nos textos fornecidos.")
        return

    # 4. Cálculo da distribuição e plotagem do gráfico
    plt.figure(figsize=(10, 6))

    # Processa e plota uma linha separada para cada termo
    for termo, posicoes in posicoes_por_termo.items():
        if not posicoes:
            continue  # Ignora termos sem ocorrência no corpus

        # Divide a extensão do texto em 20 segmentos (bins de 5% cada)
        frequencias, bordas_bins = np.histogram(posicoes, bins=20, range=(0, 100))
        # Calcula o ponto médio do eixo X para cada bin a fim de plotar a linha
        pontos_x = (bordas_bins[:-1] + bordas_bins[1:]) / 2

        plt.plot(pontos_x, frequencias, marker='o', linestyle='-', linewidth=2, label=termo)

    plt.title("Distribuição Espacial dos Termos no Corpus")
    plt.xlabel("Posição no Texto (%)")
    plt.ylabel("Frequência Absoluta")

    # Configuração dos marcadores de 10 em 10 no eixo X
    plt.xticks(np.arange(0, 101, 10))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title="Termos Alvo")

    plt.show()

# Exemplo de uso
# lista_termos = ["conclusão", "finalmente", "começo"]
# lista_textos = ["No começo nada existia. O desenvolvimento ocorreu. A conclusão finalmente chegou."]
# grafico_distribuicao_de_um_termo(lista_termos, lista_textos, lemma=True)