# Conjunto de funções básicas / abre arquivo, faz o loglikelihood, calibra o corpus loglikelihood
import nltk, os, math
from nltk import tokenize
from nltk.tokenize import word_tokenize

# Funções acessórias
def itera_ou_abre_arquivos(caminho_corpus):
    """
    Verifica se o caminho fornecidos leva a um txt, ou a diretório
    contendo arquivos txt. Retorna uma lista com os caminhos.

    Args:
        caminho_corpus(str): Caminho para um arquivo ou diretório específico.

    Returns:
        (str): O curpus é unido em uma única string
    """
    corpora = []
    # Se for um arquivo .txt, lê diretamente
    if caminho_corpus.endswith(".txt"):
        try:
            with open(caminho_corpus, "r", encoding="utf-8") as arquivo:
                corpora.append(arquivo.read())
        except OSError as e:
            print(f"Erro ao abrir o arquivo {caminho_corpus}: {e}")

    # Se for um diretório, lê todos os arquivos .txt dentro dele
    else:
        try:
            for nome_arquivo in os.listdir(caminho_corpus):
                if nome_arquivo.endswith(".txt"):
                    caminho_completo = os.path.join(caminho_corpus, nome_arquivo)
                    try:
                        with open(caminho_completo, "r", encoding="utf-8") as arquivo:
                            corpora.append(arquivo.read())
                    except OSError as e:
                        print(f"Erro ao abrir o arquivo {caminho_completo}: {e}")
        except OSError as e:
            print(f"Erro ao acessar o diretório {caminho_corpus}: {e}")
    corpora = " ".join(corpora)
    print(f"Nº Caracteres corpora: {caminho_corpus} - \n{len(corpora)}")
    return corpora

def string_to_freqdist(corpus):
    """ Transforma uma string em um objeto frqFist

    Args:
        corpus (str) - um objeto string

    Return:
        freqdist - FreqDist é uma subclasse de collections.Counter, ou seja, age como um contador de elementos.

        Métodos úteis:
        fdist.most_common(n): Retorna as n palavras mais frequentes.
        fdist.freq(palavra): Retorna a frequência relativa de uma palavra (proporção no total).
        fdist.plot(n): Plota um gráfico das n palavras mais frequentes.
        fdist.N(): Retorna o número total de palavras.
        fdist.keys(): Retorna as palavras únicas encontradas.

        Se a coleção de textos for muito grande, ele vai apresentar um resumo, como este:
        <FreqDist with 13280 samples and 67581 outcomes>
        Isso significa 13280 palavras únicas e 68581 palavras no total
     """

    # Cria o objeto token_espaco, da classe WhitespaveTokenizer,
    # Esse modulo tokenize, separa a string única em strings menores
    corpus_tokenizado = word_tokenize(corpus, language="portuguese")
    corpus_freqdist = nltk.FreqDist(word.lower() for word in corpus_tokenizado)

    return corpus_freqdist

def log_likelihood(oc, og, n, ng):
    """ Retorna um inteiro com o valor do loglikelihood

    Args:
        OC (int) - Ocorrencias no Corpus específico
        OG (int) - Ocorrencias no corpus Geral
        N  (int) - Tamanho do corpus especifico
        NG (int) - Tamanho do corpus geral

    Return - Int: resultado da verosemelhança

        Quanto maior o resultado retornado, menor a semelhança na distribuição das ocorrências.
        0, por exemplo, significaria uma distribuição exatamante igual.
        Esta função é importante para a função corpus_calibrado_log_likelihood()
    """

    taxa_ocorrencias = og / ng
    # OE - Ocerrencias esperadas com base no corpus geral
    oe = 1 if taxa_ocorrencias * n <= 0 else taxa_ocorrencias * n
    likelihood = 2 * (oc * math.log(oc / oe) + (n - oc) * (math.log((n - oc) / (n - oe))))

    return likelihood

def corpus_calibrado_log_likelihood(c_especifico_freqDist, c_geral_freqDist, limiar, stopwords=["a", "e", "i", "o", "u", ",", "?", "!"]):
    """ Entram dois corpora com freq e dist e sai um corpus com freq dist já calibrado

    Args
    # c_especifico_freqDist [freqDist] - Objeto contendo o Corpus específico
    # c_geral_freqDist [freqDist] - Objeto contendo uma amostra geral da língua
    # limiar [int] - Tolerância sobre o grau de semelhança. 0 = Mesma proporção. Bons valores variam de 1250 a 1750
    # stropwords [list] - Lista de palavras que deve ser excluidas do corpus de saída, independente do grau de varosssemelhança.

    Return - Um objeto freqdist

    A função compara o valor do loglikelihood, usando a função log_likelihood(), das ocorrencias de cada palavra do corpus específico no corpus geral.
    Se o valor estiver a baixo do limiar, a palavra é excluida da amostra.
    """

    c_especifico_calibrado = {}
    for palavra, freq in c_especifico_freqDist.items():
        if log_likelihood(freq, c_geral_freqDist[palavra], c_especifico_freqDist.N(),
                          c_geral_freqDist.N()) >= limiar and palavra not in stopwords:
            c_especifico_calibrado[palavra] = freq  # Corrigindo a atribuição ao dicionário

    c_especifico_calibrado = nltk.FreqDist(c_especifico_calibrado)
    return c_especifico_calibrado

# Função principal
# Função que egloba todas as funções da anterior. Entra uma string e retorna uma objeto freqDist (calibrado).
def lista_de_mais_frequentes_calibrada(caminho_corpus_especifico, caminho_corpus_geral, limiar, raw = True, stopwords=["a", "e", "i", "o", "u", ",", "?", "!"]):
    """
        Abre duas str com o local dos corpora, sejam arquivos ou diretórios
        processa e devolve um arquivo FreqDist já com as palavras que estão acima do limiar retiradas.

    Args:
        caminho_corpus_especifico (str): Caminho para a pasta ou para o txt com o corpus específico
        caminho_corpus_geral (str): Caminho para a pasta ou para o txt com o corpus específico
        limiar (int):
        raw (bool): Usado quando a entrada da função não é o caminho para o arquivo do corpus, mas o corpus em si.
                    Dever ser marcado como False quando a entrada da função já for a string que será analisada.
        stopwords (list):

    return:
        (nltk.probability.FreqDist): Um objeto freqdist do corpus específico já calibrado na probabilidade.

        Métodos úteis:
        fdist.most_common(n): Retorna as n palavras mais frequentes.
        fdist.freq(palavra): Retorna a frequência relativa de uma palavra (proporção no total).
        fdist.plot(n): Plota um gráfico das n palavras mais frequentes.
        fdist.N(): Retorna o número total de palavras.
        fdist.keys(): Retorna as palavras únicas encontradas.

    A função compara o valor do loglikelihood, usando a função log_likelihood(), das ocorrencias de cada palavra do corpus específico no corpus geral.
    Se o valor estiver a baixo do limiar, a palavra é excluida da amostra.

     A função Depende de várias funções anteriores, como:
     itera_ou_abre_arquivos()
     string_to_freqdist()
     log_likelihood()
     corpus_calibrado_log_likelihood()
    """

    # Carrega o corpus específico (string ou arquivo)
    if raw:
        corpus_especifico = itera_ou_abre_arquivos(caminho_corpus_especifico)
    else:
        corpus_especifico = caminho_corpus_especifico

    # Sempre carrega o corpus geral a partir de arquivo(s)
    corpus_geral = itera_ou_abre_arquivos(caminho_corpus_geral)

    # String to freqdist
    c_especifico_tokenizado = string_to_freqdist(corpus_especifico)
    c_geral_tokenizado = string_to_freqdist(corpus_geral)

    # Chama a função corpus freqdist calibrado
    corpus_calibrado = corpus_calibrado_log_likelihood(c_especifico_tokenizado, c_geral_tokenizado, limiar, stopwords)
    return corpus_calibrado