import spacy

# Carregamento do modelo linguístico
try:
    nlp = spacy.load("pt_core_news_sm")
except OSError:
    raise OSError("Modelo linguístico não encontrado. Execute: python -m spacy download pt_core_news_sm")


def conta_tags_morfologicas_e_sintaticas(termo, lista_de_setencas, lemma=True):
    """
    Essa função conta quantas vezes e quais papeis sintáticos e morfológicos
    um termo exerce em uma lista de setencas.

    1 - A função itera um conjunto de setencas em uma lista buscando o termo alvo
    2 - Caso encontre, a função usar o parse da biblioteca Spacy para atribuir tags sintáticas e semânticas ao termo alvo
    3 - A função salva as informações em um dicionário.
        - A chave do dicionario é composta pelo nome duas tags. Exemplo: NOUN | obj
        - O valor da chave é composto por uma tupla. O primeiro valor da tupla é um int com númeto de vezes que
        o termo aparece com a tags da chave. E seguno valor da tupla é uma lista com as orações onde o termo aparece
        com as tagas da chave.
    4 - A função retora o dicionáio e também um print com uma tabela mostrando a contagem (primeiro valore da tupla), a chave,
    e a primeira frase (primero item do dicionário que é o segunda entrada da tupla)

    :argumentos:

    termo(str): termo alvo que será buscado nas sentenças
    lista_de_setencas(list): lista de setencas
    lemma(bool): Define se a busca deve considerar tanto o termo alvo como os termos das sentenças em
    sua versão lematizada (lexema).

    :returns
    dic{str:(int,list[str]}:
    """

    # Processamento do termo alvo
    alvo = termo.lower()
    if lemma:
        doc_alvo = nlp(alvo)
        alvo = doc_alvo[0].lemma_.lower()

    resultados = {}
    total_sentencas = len(lista_de_setencas)

    # Varredura das sentenças utilizando pipe para maior eficiência
    for idx, doc in enumerate(nlp.pipe(lista_de_setencas)):
        # Marcador de progresso
        if total_sentencas > 0 and (idx + 1) % max(1, total_sentencas // 100) == 0:
            progresso = ((idx + 1) / total_sentencas) * 100
            print(f"\rProcessando sentenças: {progresso:.1f}% concluído [{idx + 1}/{total_sentencas}]", end="",
                  flush=True)

        for token in doc:
            forma_analisada = token.lemma_.lower() if lemma else token.text.lower()

            # Verificação de correspondência do termo
            if forma_analisada == alvo:
                chave = f"{token.pos_} | {token.dep_}"

                # Inicialização da chave no dicionário, caso não exista
                if chave not in resultados:
                    resultados[chave] = (0, [])

                # Atualização da contagem e da lista de orações
                contagem, lista_sents = resultados[chave]
                lista_sents.append(doc.text)
                resultados[chave] = (contagem + 1, lista_sents)

    print("\n")  # Quebra de linha após a conclusão da barra de progresso

    # Impressão da tabela
    print(f"{'Contagem':<10} | {'Tag (POS | DEP)':<20} | {'Primeira Sentença'}")
    print("-" * 80)

    # Ordenação decrescente pelo número de ocorrências (contagem)
    resultados_ordenados = sorted(resultados.items(), key=lambda item: item[1][0], reverse=True)

    for chave, (contagem, lista_sents) in resultados_ordenados:
        primeira_sentenca = lista_sents[0] if lista_sents else ""
        print(f"{contagem:<10} | {chave:<20} | {primeira_sentenca.strip()}")

    return resultados

# Exemplo de uso:
# sentencas = ["O estado forte garante os direitos.", "Sem o estado, não há garantias.", "Ele foi processado pelo estado."]
# dict_resultados = conta_tags_morfologicas_e_sintaticas("estado", sentencas, lemma=True)