import json

def carregar_json_para_dicionario(caminho_arquivo):
    """
    Abre um arquivo JSON e retorna seu conteúdo como um dicionário.
    """
    try:
        with open(caminho_arquivo, 'r', encoding='utf-8') as arquivo:
            dados = json.load(arquivo)
        return dados
    except FileNotFoundError:
        print(f"❌ Erro: O arquivo '{caminho_arquivo}' não foi encontrado.")
        return None
    except json.JSONDecodeError:
        print(f"❌ Erro: O arquivo '{caminho_arquivo}' não é um JSON válido.")
        return None
    except Exception as e:
        print(f"❌ Ocorreu um erro inesperado: {e}")
        return None