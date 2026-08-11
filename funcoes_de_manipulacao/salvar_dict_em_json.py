import json


def salvar_dict_em_json(dados, caminho_arquivo, indent=4):
    """
    Salva um dicionário em um arquivo JSON (UTF-8, com acentos preservados).

    Args:
        dados (dict): Dicionário a ser salvo (ex.: {titulo: {tema: valor}}).
        caminho_arquivo (str): Caminho do arquivo .json de saída.
        indent (int | None): Indentação para deixar o JSON legível. Use None para
            gravar em uma única linha (mais compacto). Padrão: 4.

    Returns:
        str: O caminho do arquivo gravado.

    Observações:
        - `ensure_ascii=False` mantém acentos ("Religião" em vez de "Religi\\u00e3o").
        - Tipos numéricos do numpy (np.float64, np.int64) são convertidos para os tipos
          nativos do Python, evitando erro de serialização.
    """
    def _conversor(obj):
        # Fallback para tipos não serializáveis nativamente (ex.: numpy)
        if hasattr(obj, "item"):
            return obj.item()
        raise TypeError(f"Tipo não serializável em JSON: {type(obj).__name__}")

    with open(caminho_arquivo, "w", encoding="utf-8") as arquivo:
        json.dump(dados, arquivo, ensure_ascii=False, indent=indent, default=_conversor)

    print(f"Dicionário salvo em: {caminho_arquivo}")
    return caminho_arquivo
