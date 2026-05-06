# Conjunto de funções básicas / abre arquivo, faz o loglikelihood, calibra o corpus loglikelihood
import nltk, os, math
from nltk.tokenize import word_tokenize

stop_words_pt = [
    "ó","né",'aí',"isso",".... ","..","...","|","?","!","vc""né", "gente",
    "então", "tá", "ai", "só", ",", "eh", ".", 'a', 'abaixo', 'acerca', 'acima', 'ademais', 'adeus', 'agora', 'ainda',
    'algo', 'algumas', 'alguns', 'ali', 'ambos', 'antes', 'ao', 'aos', 'aonde',
    'apenas', 'apesar', 'após', 'aquela', 'aquelas', 'aquele', 'aqueles',
    'aquilo', 'aqui', 'as', 'assim', 'até', 'através', 'atrás', 'c', 'cada',
    'catorze', 'cedo', 'cem', 'cento', 'certamente', 'certeza', 'cinco',
    'coisa', 'com', 'como', 'conosco', 'consoante', 'consta', 'contigo',
    'contudo', 'contra', 'cuja', 'cujas', 'cujo', 'cujos', 'd', 'da', 'daquela',
    'daquelas', 'daquele', 'daqueles', 'das', 'de', 'debaixo', 'demais',
    'demasiado', 'dentro', 'depois', 'desde', 'dessa', 'dessas', 'desse',
    'desses', 'desta', 'destas', 'deste', 'destes', 'deve', 'devem', 'devendo',
    'dever', 'deverá', 'deverão', 'deveria', 'deveriam', 'devia', 'deviam',
    'dez', 'dezanove', 'dezasseis', 'dezassete', 'dezoito', 'diante', 'disso',
    'disto', 'dito', 'diz', 'dizem', 'dizer', 'do', 'dois', 'donde', 'dos',
    'doze', 'duas', 'duzentos', 'durante', 'e', 'é', 'ela', 'elas', 'ele',
    'eles', 'em', 'embora', 'enquanto', 'entre', 'era', 'eram', 'éramos',
    'és', 'essa', 'essas', 'esse', 'esses', 'esta', 'está', 'estamos',
    'estando', 'estar', 'estará', 'estaremos', 'estaria', 'estariam',
    'estaríamos', 'estava', 'estávamos', 'este', 'esteja', 'estejam',
    'estejamos', 'estes', 'esteve', 'estive', 'estivemos', 'estiver', 'estivera',
    'estiveram', 'estivéramos', 'estiverem', 'estivermos', 'estivesse',
    'estivessem', 'estivéssemos', 'estiveste', 'estivestes', 'estou', 'eu',
    'excepto', 'f', 'faço', 'falta', 'favor', 'faz', 'fazeis', 'fazem',
    'fazemos', 'fazer', 'fazes', 'feita', 'feitas', 'feito', 'feitos', 'fez',
    'fim', 'final', 'fiz', 'fizer', 'fizeram', 'fizerem', 'fizermos', 'fizesse',
    'fizéssemos', 'foi', 'fomos', 'for', 'fora', 'foram', 'fôramos', 'forem',
    'formos', 'fosse', 'fossem', 'fôssemos', 'foste', 'fostes', 'fui', 'g', 'h',
    'há', 'haja', 'hajam', 'hajamos', 'hão', 'havemos', 'havendo', 'haver',
    'haverá', 'haverão', 'haverei', 'haveremos', 'haveria', 'haveriam',
    'haveríamos', 'havia', 'haviam', 'havíamos', 'hei', 'haja', 'houve',
    'houvemos', 'houver', 'houvera', 'houveram', 'houvéramos', 'houverão',
    'houverei', 'houverem', 'houveremos', 'houveria', 'houveriam', 'houveríamos',
    'houvermos', 'houvesse', 'houvessem', 'houvéssemos', 'i', 'j', 'já',
    'jamais', 'junto', 'l', 'lá', 'lado', 'lhe', 'lhes', 'logo', 'longe',
    'm', 'maior', 'maioria', 'mais', 'mal', 'mas', 'me', 'mediante', 'meio',
    'melhor', 'menos', 'mesma', 'mesmas', 'mesmo', 'mesmos', 'meu', 'meus',
    'mil', 'minha', 'minhas', 'momento', 'muito', 'muitos', 'n', 'na', 'nada',
    'não', 'naquela', 'naquelas', 'naquele', 'naqueles', 'nas', 'nem', 'nenhum',
    'nenhuma', 'nessa', 'nessas', 'nesse', 'nesses', 'nesta', 'nestas', 'neste',
    'nestes', 'ninguém', 'no', 'noite', 'nome', 'nos', 'nós', 'nossa', 'nossas',
    'nosso', 'nossos', 'nova', 'novas', 'nove', 'novo', 'novos', 'num', 'numa',
    'nunca', 'o', 'oitava', 'oitavo', 'oito', 'onde', 'ontem', 'onze', 'os',
    'ou', 'outra', 'outras', 'outrem', 'outro', 'outros', 'p', 'para', 'parece',
    'parte', 'partir', 'pela', 'pelas', 'pelo', 'pelos', 'per', 'perante',
    'perto', 'pode', 'podem', 'podendo', 'poder', 'poderá', 'poderão', 'poderia',
    'poderiam', 'podia', 'podiam', 'põe', 'põem', 'pois', 'ponto', 'pontos',
    'por', 'porém', 'porque', 'porquê', 'portanto', 'posição', 'possível',
    'posso', 'pouca', 'poucas', 'pouco', 'poucos', 'pra', 'própria', 'próprias',
    'próprio', 'próprios', 'próxima', 'próximas', 'próximo', 'próximos', 'q',
    'quais', 'qual', 'qualquer', 'quando', 'quanta', 'quantas', 'quanto',
    'quantos', 'quão', 'que', 'quê', 'quem', 'quer', 'quereis', 'querem',
    'queremas', 'queres', 'quero', 'questão', 'quiçá', 'quinta', 'quinto', 'r',
    'relação', 's', 'sabe', 'sabem', 'são', 'se', 'seja', 'sejam', 'sejamos',
    'sem', 'sempre', 'sendo', 'ser', 'será', 'serão', 'serei', 'seremos',
    'seria', 'seriam', 'seríamos', 'sete', 'sétima', 'sétimo', 'seu', 'seus',
    'sexta', 'sexto', 'si', 'sido', 'sim', 'sob', 'sobre', 'sois', 'somos',
    'sou', 'sua', 'suas', 't', 'tão', 'tal', 'talvez', 'também', 'tampouco',
    'tanta', 'tantas', 'tanto', 'tarde', 'te', 'tem', 'têm', 'temos', 'tendo',
    'tenha', 'tenham', 'tenhamos', 'tenho', 'tens', 'ter', 'terá', 'terão',
    'terceira', 'terceiro', 'terei', 'teremos', 'teria', 'teriam', 'teríamos',
    'teu', 'teus', 'teve', 'ti', 'tido', 'tinha', 'tinham', 'tínhamos', 'tive',
    'tivemos', 'tiver', 'tivera', 'tiveram', 'tivéramos', 'tiverem', 'tivermos',
    'tivesse', 'tivessem', 'tivéssemos', 'tiveste', 'tivestes', 'toda', 'todas',
    'todavia', 'todo', 'todos', 'três', 'treze', 'tu', 'tua', 'tuas', 'tudo',
    'u', 'um', 'uma', 'umas', 'uns', 'usa', 'usar', 'v', 'vai', 'vais', 'vão',
    'várias', 'vários', 'vem', 'vêm', 'vendo', 'ver', 'vez', 'vezes', 'viagem',
    'vindo', 'vinte', 'vir', 'virá', 'virão', 'você', 'vocês', 'vos', 'vós',
    'vossa', 'vossas', 'vosso', 'vossos', 'z', 'zero', 'à', 'às', 'área'
]


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
    print(f"Nº Caracteres corpora: {caminho_corpus} - {len(corpora)}")
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

# Função que egloba todas as funções da anterior. Entra uma string e retorna uma objeto freqDist (calibrado).
def lista_de_mais_frequentes_calibrada(caminho_corpus_especifico, caminho_corpus_geral, limiar, raw = True, stopwords=stop_words_pt):
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
    print(f'Tamanho do corpus específico: {len(corpus_especifico)}')
    # Sempre carrega o corpus geral a partir de arquivo(s)
    corpus_geral = itera_ou_abre_arquivos(caminho_corpus_geral)
    print(f'Tamanho do corpus geral: {len(corpus_geral)}')

    # String to freqdist
    c_especifico_tokenizado = string_to_freqdist(corpus_especifico)
    c_geral_tokenizado = string_to_freqdist(corpus_geral)

    # Chama a função corpus freqdist calibrado
    corpus_calibrado = corpus_calibrado_log_likelihood(c_especifico_tokenizado, c_geral_tokenizado, limiar, stopwords)
    corpus_calibrado = corpus_calibrado_log_likelihood(c_especifico_tokenizado, c_geral_tokenizado, limiar, stopwords)
    return corpus_calibrado

