import spacy

# Carregamento do modelo
nlp = spacy.load("pt_core_news_sm")

def segmentar_sentencas(texto):
    """
    Recebe uma string e retorna uma lista de sentenças utilizando o motor do spaCy.
    """
    # Desabilita componentes desnecessários para aumentar a velocidade (NER, Parser complexo)
    # Mantém apenas o 'senter' (Sentence Recognizer) se disponível ou o parser básico.
    doc = nlp(texto)
    return [sent.text.strip() for sent in doc.sents]

# Exemplo de uso
texto_exemplo = "O Dr. Nikolas discursou na Avenida Paulista. O evento ocorreu em março de 2024!"
resultado = segmentar_sentencas(texto_exemplo)
print(resultado)

def segmentar_sentencas_em_lote(lista_de_textos):
    sentencas_totais = []
    # n_process=-1 utiliza todos os núcleos do processador (Multiprocessing)
    i = 0
    for doc in nlp.pipe(lista_de_textos, batch_size=50):
        for sent in doc.sents:
            sentencas_totais.append(sent.text.strip())
        i += 1
        print(f'Processando Lote {i}')
    return sentencas_totais