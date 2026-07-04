import spacy

# Carregamento do modelo linguístico em português
try:
    nlp = spacy.load("pt_core_news_sm")
except OSError:
    raise OSError("Modelo não encontrado. Execute: python -m spacy download pt_core_news_sm")


def extrair_relacoes_sintaticas(texto: str):
    """
    Executa a análise de dependência sintática (parsing) no texto fornecido.
    Imprime as relações entre núcleo (head), dependente (child) e o rótulo da dependência.
    """
    doc = nlp(texto)

    print(f"{'Token':<15} | {'Rótulo (dep_)':<15} | {'Núcleo (head)':<15}")
    print("-" * 50)

    for token in doc:
        # Extrai o texto do token, a sua etiqueta de dependência e o texto do seu nó central (pai)
        print(f"{token.text:<15} | {token.dep_:<15} | {token.head.text:<15}")

    print("\nExtração de Sujeitos e Objetos do ROOT:")
    for token in doc:
        if token.dep_ == "ROOT":
            print(f"Verbo Principal (ROOT): {token.text}")
            for filho in token.children:
                if filho.dep_ in ("nsubj", "nsubj:pass"):
                    print(f" -> Sujeito: {filho.text}")
                elif filho.dep_ in ("obj", "iobj"):
                    print(f" -> Objeto: {filho.text}")
