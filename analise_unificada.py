import pandas as pandas
import numpy as np
import statsmodels.api as sm

import pandas as pd
import numpy as np
import statsmodels.api as sm

def rodar_regressao(df, nome_estudo, var_y, var_x, nomes_x):
    """
    Roda a regressão OLS em uma base de dados e exibe os resultados organizados.
    """
    # Remover linhas com valores nulos
    dados = df.dropna(subset=[var_y] + var_x).copy()
    
    # Adicionar constante (intercepto \beta_0)
    X = sm.add_constant(dados[var_x])
    y = dados[var_y]
    
    # Ajustar o modelo de regressão linear OLS
    modelo = sm.OLS(y, X).fit()
    
    print("\n" + "="*60)
    print(f" RESULTADOS DA REGRESSÃO: {nome_estudo.upper()}")
    print("="*60)
    print(f"Número de Observações (N): {int(modelo.nobs)}")
    print(f"R² (Poder Explicativo): {modelo.rsquared:.4f} ({modelo.rsquared*100:.2f}%)")
    print(f"R² Ajustado: {modelo.rsquared_adj:.4f}")
    print("-" * 60)
    
    # Tabela de coeficientes simplificada e legível
    tabela_coef = pd.DataFrame({
        'Coeficiente (Beta)': modelo.params,
        'Erro-Padrão': modelo.bse,
        'Estatística t': modelo.tvalues,
        'p-valor': modelo.pvalues
    })
    
    # Renomear os índices para o gráfico/tabela ficarem legíveis
    renomear_map = {'const': 'Intercepto (Beta 0)'}
    for orig, novo in zip(var_x, nomes_x):
        renomear_map[orig] = novo
    tabela_coef.index = [renomear_map.get(col, col) for col in tabela_coef.index]
    
    print(tabela_coef.to_string())
    print("="*60)
    
    return modelo


# ==============================================================================
# 1. CARREGAMENTO E LIMPEZA DA BASE 1: PNAS (Creswell et al.)
# ==============================================================================
print("\n[1/2] A carregar e a tratar a base da PNAS (Creswell et al.)...")
df_pnas = pd.read_csv('pnas.2209123120.sd01 (1) (2).csv', sep=';')

# Função para corrigir strings corrompidas com múltiplos pontos nos decimais
def fix_float(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    parts = s.split('.')
    if len(parts) > 2:
        s = parts[0] + '.' + ''.join(parts[1:])
    try:
        return float(s)
    except:
        return np.nan

# Tratar tempo de sono e GPA
df_pnas['TotalSleepTime_clean'] = df_pnas['TotalSleepTime'].apply(fix_float)
df_pnas['term_gpa_clean'] = df_pnas['term_gpa'].apply(fix_float)
df_pnas['demo_gender_clean'] = pd.to_numeric(df_pnas['demo_gender'], errors='coerce')

# Converter minutos para horas de sono
df_pnas['sono_horas'] = df_pnas['TotalSleepTime_clean'] / 60.0

# Filtrar observações com GPA válido (entre 0 e 4)
df_pnas_valid = df_pnas[
    (df_pnas['term_gpa_clean'] >= 0) & 
    (df_pnas['term_gpa_clean'] <= 4.0)
].copy()


# ==============================================================================
# 2. CARREGAMENTO E LIMPEZA DA BASE 2: PEERJ
# ==============================================================================
print("[2/2] A carregar e a tratar a base da PeerJ...")
excel_path = 'peerj-09-11107-s002.xlsx'

df_peerj_sleep = pd.read_excel(excel_path, sheet_name='Sleep')
df_peerj_demo = pd.read_excel(excel_path, sheet_name='Demographics')

# Concatenar as informações das abas
df_peerj = pd.concat([df_peerj_sleep, df_peerj_demo['Biological sex']], axis=1)

# Mapear faixas categóricas de horas de sono para valores contínuos (ponto médio)
mapping_sleep = {
    '4 hours or less': 4.0,
    '5 hours': 5.0,
    '6 hours': 6.0,
    '7 hours': 7.0,
    '8 hours': 8.0,
    '9 hours': 9.0,
    '10 or more hours': 10.0
}

df_peerj['sono_horas'] = df_peerj['Hours of sleep'].map(mapping_sleep)
df_peerj['gpa_clean'] = pd.to_numeric(df_peerj['Current GPA'], errors='coerce')

# Variable dummy: 1 se for Mulher (Female), 0 caso contrário
df_peerj['gender_female'] = (df_peerj['Biological sex'] == 'Female').astype(int)

# Filtrar observações com GPA válido (entre 0 e 4)
df_peerj_valid = df_peerj[
    (df_peerj['gpa_clean'] >= 0) & 
    (df_peerj['gpa_clean'] <= 4.0)
].copy()


# ==============================================================================
# 3. EXECUÇÃO DAS REGRESSÕES E COMPARAÇÃO
# ==============================================================================

# Executar para a PNAS (Creswell)
mod_pnas = rodar_regressao(
    df=df_pnas_valid,
    nome_estudo="PNAS (Creswell et al.)",
    var_y='term_gpa_clean',
    var_x=['sono_horas', 'demo_gender_clean'],
    nomes_x=['Horas de Sono', 'Género (1 = Feminino / Outro)']
)

# Executar para a PeerJ
mod_peerj = rodar_regressao(
    df=df_peerj_valid,
    nome_estudo="PeerJ",
    var_y='gpa_clean',
    var_x=['sono_horas', 'gender_female'],
    nomes_x=['Horas de Sono', 'Sexo Biológico (1 = Feminino)']
)

print("\nAnálise concluída com sucesso!")