#!/usr/bin/env python
# coding: utf-8

# # Operações de Simulação com o Servidor ASA

# Esta célula importa todos os módulos necessários para execução e análise das simulações.

# In[48]:


import asapy  # Fornece vinculações Python para a API AsaPy
import asaclient # O módulo 'asaclient' é usado para estabelecer uma interface de cliente Python com o servidor Asa.import pandas as pd
import pandas as pd
import re
from asapy.preprocessing import Preprocessing
from asapy.preprocessing import AsaType, AsaCustomType
import numpy as np


# ## Login e Autenticação <a name="login"></a>
# 
# A primeira etapa envolve login e autenticação, necessários para o usuário acessar o sistema. Inicializamos uma instância da classe `ASA` do módulo `asaclient`, definindo o parâmetro do servidor para a URL especificada. Esta ação estabelece uma conexão com o Servidor ASA.

# In[50]:


asa = asaclient.ASA(server="https://service.asa.dcta.mil.br")


# 
# ## Carregando e Salvando Simulações <a name="carregando_simulacoes"></a>
# 
# Nesta seção, carregamos manualmente uma simulação a partir de um arquivo JSON. Este processo envolve abrir o arquivo JSON, carregar seu conteúdo em um dicionário Python e usar esses dados para instanciar um objeto de simulação.

# In[51]:


simulation_name = "Wind_seed_variavel"
sim = asapy.load_simulation(f"./{simulation_name}.json")  # Creating a Simulation from the JSON file at the specified path, and assigning it to 'sim'.
sim = asa.save_simulation(sim)
sim


# ## Listagem de Aliases <a name="listagem_aliases"></a>
# 
# Esta seção envolve a enumeração de todos os aliases presentes em uma simulação.
# 
# Primeiro instanciamos um objeto da classe ``Doe`` (Design of Experiments). Em seguida, processamos os aliases (parâmetros utilizados nas simulações) para a simulação escolhida ``sim``, usando o método ``process_aliases_by_sim`` e passando as configurações dos componentes do servidor ASA. Os aliases processados são armazenados na variável ``aliases`` e exibidos. A função ``asa.component_configs()`` retorna uma lista de configurações de componentes. Essa informação é crucial para entender os números padrão, mínimo e máximo de componentes em um determinado cenário.

# In[52]:


doe = asapy.Doe()
aliases = doe.process_aliases_by_sim(sim, asa.component_configs()) 
import pandas as pd
pd.set_option('display.max_rows', None)
aliases


# ## Edição dos Aliases <a name="edicao_aliases"></a>
# 
# Esta seção permite que os parâmetros dos aliases listados sejam modificados para atender às suas necessidades específicas.
# 
# Esta operação apresenta todas as alias (aleatoriedades) estão listadas.

# In[53]:


# Filtrar somente as linhas onde o tipo é 'double'
double_aliases = aliases[aliases['type'] == 'double']

# Criar a string de conditions com os valores de min e max corretos
conditions = "conditions = [\n"
for label, row in double_aliases.iterrows():
    conditions += f"    ('{label}', {row['min']}, {row['max']}),\n"
conditions += "]"

print(conditions)


# ## Edição dos Aliases <a name="edicao_aliases"></a>
# 
# Esta seção permite que os parâmetros dos aliases listados sejam modificados para atender às suas necessidades específicas.
# 
# Esta operação apresenta todas as alias (aleatoriedades) estão listadas.

# In[54]:


aliases_reset = aliases.reset_index()
nomes_aliases = aliases_reset.iloc[:, 0]  # Isso pega todos os valores da primeira coluna

codigo = ''
for nome in nomes_aliases:
    codigo += f"aliases.loc['{nome}', 'min'] = \n"
    codigo += f"aliases.loc['{nome}', 'max'] = \n\n"

print(codigo)


# ### Ajuste Manual 
# Como pode se ajustar manualmente o intervalo de um parâmetro de alias dentro do seu modelo de lote.

# In[55]:


aliases.loc['WEZ_ratio', 'min'] = 0.2
aliases.loc['WEZ_ratio', 'max'] = 1

aliases.loc['GBAD_reload_time', 'min'] = 6
aliases.loc['GBAD_reload_time', 'max'] = 20

aliases.loc['wind_ang', 'min'] = 75
aliases.loc['wind_ang', 'max'] = 90

aliases.loc['wind_speed_factor', 'min'] = 1
aliases.loc['wind_speed_factor', 'max'] = 2

aliases.loc['wind_tempo_reta', 'min'] = 10
aliases.loc['wind_tempo_reta', 'max'] = 20

aliases


# ## Criando o design dos experimentos (DOE) <a name="criando_doe"></a>
# 
# Os usuários têm a capacidade de estabelecer um DOE, que lhes permite planejar, conduzir e analisar um conjunto de experimentos de maneira sistemática.
# 
# Nesta célula, estamos usando o método 'create' do objeto 'Doe' para gerar uma tabela de Design de Experimentos (DOE). O método recebe como argumentos os aliases processados anteriormente e o número de amostras. O DOE resultante é um DataFrame do pandas, que é armazenado na variável 'df' e então exibido.
# O DOE é criado usando um método de Amostragem de Hipercubo Latino (LHS) e um tamanho de amostra samples. A função retorna um novo DataFrame com os nomes e valores das variáveis de entrada de acordo com o seu tipo.

# In[56]:


df = doe.create(aliases, samples=450, seed=11)
df


# ## Listando Métricas separadas para a avaliação do experimento <a name="listando_metricas"></a>
# 
# Esta etapa envolve a geração de uma lista de todas as métricas usadas em uma simulação.
# 
# O seguinte segmento de código recupera e processa as métricas associadas à simulação ``sim``. O método ``process_metrics`` do objeto ``Doe`` é usado para obter um DataFrame, ``metrics_df``, contendo os dados das métricas. A linha de código subsequente seleciona a linha rotulada como ``Team_Metrics_Blue`` do DataFrame e recupera as métricas correspondentes usando a coluna ``metrics``.

# In[57]:


metrics_df = doe.process_metrics(sim)
metrics_df


# ## Inspecionando especificando as métricas 
# Analisando as do time `blue` para ter uma idéia do todo

# In[58]:


print(metrics_df.loc['Team_Metrics_Blue']['metrics'])


# ## Criando uma função de Parada para a execução do Lote 

# A variável `metric` é atribuída ao valor da string `acft_standing`, representando uma métrica específica de interesse no contexto da análise. Esta métrica será importante ao implementar o recurso de parada automática durante a execução em lote.
# 

# In[59]:


metric = 'acft_standing' 
side= 'blue'


# In[60]:


#stop_func=asapy.non_stop_func
#stop_func=asapy.stop_func(metric=metric, threshold=0.000001, side=side)


# ## Executando um Lote com Parada Automática e Barra de Progresso <a name="executando_lote"></a>
# 

# ### Cria o batch 
# O lote está associado à simulação identificada por ``sim.id``. Após sua criação, ele é enviado para o sistema para posterior processamento e eventual execução.
# 

# In[61]:


batch = asaclient.Batch(label=simulation_name, simulation_id=sim.id)


# ### Salva o batch no serviço ASA
# Este recurso permite aos usuários executar um processo em lote e uma barra de progresso para acompanhar visualmente sua conclusão.

# In[62]:


batch = asa.save_batch(batch)
print(f"Batch criado: {batch.id}")


# ### Cria o objeto ExecutionControler, responsável por gerenciar a execução do Batch.
# 
# Os códigos a seguir inicializa um objeto `ExecutionController` chamado `ec`. O parâmetro `sim_func` é definido como `asapy.batch_simulate(batch=batch)`, que especifica a função responsável pela execução da simulação em lote.
# 
# `asa::recorder::AsaMonitorReport` traz os relatórios de monitoramento para a simulação, ou seja, as métricas utilizadas para controlar a execução do lote estão listadas.
# 
# O parâmetro `stop_func` é definido como `asapy.stop_func(metric=metric, threshold=0.001)`, que indica a função a ser usada para parar a execução com base em uma métrica e um limite especificados.
# 
# O parâmetro `chunk_size` determina o tamanho de cada pedaço de simulações a ser executado simultaneamente.

# In[63]:


ec = asapy.ExecutionController(sim_func=asapy.batch_simulate(batch=batch), stop_func=asapy.non_stop_func, chunk_size=10)


# ### Adicionando as variações (aliases) ao lote 
# O método `run` do objeto `ec` é executado, passando o DataFrame `df` os com aliases. O resultado da execução é armazenado na variável `results`.

# In[64]:


get_ipython().run_cell_magic('time', '', '\nresults = ec.run(doe=df)')


# In[65]:


results.head()


# ## Excluindo as MIXR Messages e Verificando os tipos de Asa Messages disponíveis

# In[66]:


results = results[results['mixr_type'] == 1000]
print("Asa Messages: ")
print({k:AsaType(k).name for k in results['asa_type'].unique()  if not np.isnan(k)})


# ## Verifica os tipos de Custom Messages disponíveis

# In[67]:


print("Asa Custom Messages: ")
print({k: AsaCustomType(k).name for k in results['asa_custom_type'].unique() if isinstance(k, str) and len(k) > 0})


# ## Armazenando as informações em csv pra não precisar rodar o ASA toda hora

# In[68]:


simulation_name = "Trab_final_Wind_WEZ_resultado_bruto_F_A"
file_name = f'results_{simulation_name}.csv'
results.to_csv(file_name, index=False)


# ## IMPORTANDO as informações em csv pra não precisar rodar o ASA toda hora

# In[69]:


simulation_name = "Trab_final_Wind_WEZ_resultado_bruto_F_A"
file_name = f'results_{simulation_name}.csv'
imported_results = pd.read_csv(file_name)
results=imported_results
results.head()


# ## Combina a tabela de resultados com o a tabela de experimentos para vincular as variáveis de entrada (aliases) com os resultados.

# In[70]:


df['experiment'] = df.index
results = results.merge(df, on='experiment')
results.head()


# ## Verifica as colunas disponíves para selecionar os extra_fields

# In[71]:


print("Colunas disponíveis")
print([c for c in results.columns])


# In[72]:


extra_fields = ['experiment','seed','sim_time',  'WEZ_ratio', 'GBAD_reload_time', 'wind_ang', 'wind_S_ou_N', 'wind_speed_factor', 'wind_tempo_reta']
extra_fields


# ## Filtra as mensagens e converte o payload usando a função **parse_recorder**

# In[73]:


sub_df = Preprocessing.parse_recorder(
    results, asa_custom_type=AsaCustomType.TEAM_METRICS, extra_fields=extra_fields)
sub_df.head()


# In[74]:


print("Colunas disponíveis")
print([c for c in sub_df.columns])


# ## Filtra, Renomeia e reordena as colunas para facilitar a análise

# In[75]:


cols = sub_df.columns.tolist()
new_order = ['experiment', 'side', 'air_vehicle_damage', 'building_damage'] + [col for col in cols if col not in ['experiment', 'side', 'air_vehicle_damage', 'building_damage']]

sub_df = sub_df[new_order]

sub_df.rename(columns={
    'experiment': 'exp'
}, inplace=True)
sub_df.head()


# ## Filtro dos momentos em que acontecem as alterações desejadas

# In[76]:


def extract_changes(df, columns):
    # Verifica se as colunas especificadas estão no DataFrame
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"A coluna '{col}' não está presente no DataFrame.")
    
    # Calcula a diferença entre cada elemento e o anterior
    diffs = df[columns].diff()
    
    # Encontra as primeiras linhas onde ocorre uma mudança
    change_indices = diffs[(diffs != 0) & (~diffs.isna())].dropna(how='all').index
    
    # Retorna as linhas do DataFrame original correspondentes aos índices de mudança
    return df.loc[change_indices]


# In[77]:


df_clean = extract_changes(sub_df,['acft_killed','building_damage'])
def rename_columns(df):
    df.columns = df.columns.str.replace('.', '_', regex=False)
    return df
rename_columns(df_clean)
df_clean.head(10)


# ## Filtra as colunas desejadas - Reordenação e organização do Data Frame

# In[78]:


def extract_and_sort(df):
    # Filtra as colunas desejadas
    columns_to_keep = ['exp', 'side', 'sam_hit','sam_lost', 'sam_remaining', 'acft_killed', 
                       'acft_standing', 'bmb_released', 'building_damage', 'sim_time',  'WEZ_ratio',
                       'GBAD_reload_time', 'wind_ang', 'wind_S_ou_N', 'wind_speed_factor', 'wind_tempo_reta']
    df_filtered = df[columns_to_keep]
    
    # Define as colunas a serem verificadas para mudanças
    columns_to_check = ['building_damage', 'acft_damaged']
    
    # Calcula a diferença entre cada elemento e o anterior
    diffs = df[columns_to_check].diff()
    
    # Encontra as primeiras linhas onde ocorre uma mudança
    change_indices = diffs[(diffs != 0) & (~diffs.isna())].dropna(how='all').index
    
    # Retorna as linhas do DataFrame filtrado correspondentes aos índices de mudança
    df_changes = df_filtered.loc[change_indices]
    
    # Define a ordem desejada para a coluna 'side'
    side_order = {'blue': 0, 'red': 1}
    
    # Cria uma coluna temporária para a ordem do 'side'
    df_changes['side_order'] = df_changes['side'].map(side_order)
    
    # Ordena o DataFrame pela coluna 'exp' e depois pela coluna 'side_order'
    df_sorted = df_changes.sort_values(by=['exp', 'side_order'])
    
    # Remove a coluna temporária
    df_sorted = df_sorted.drop(columns=['side_order'])
    
    return df_sorted


# In[79]:


def extract_time(sim_time):
    return sim_time[14:-6]  # Extrai os caracteres do 14º ao penúltimo (remove '1970-01-01T09:' do início e '-03:00' do final)


# In[80]:


df_clean_2 = extract_and_sort(df_clean)
df_clean_2['sim_time'] = df_clean_2['sim_time'].apply(extract_time)
df_clean_2.head()


# ## Criando um Data Frame resumido

# In[81]:


import pandas as pd

# Função para criar o sumário
def create_summary(df):
    summary_data = []

    grouped = df.groupby('exp')

    for exp, group in grouped:
        red_group = group[group['side'] == 'red']
        blue_group = group[group['side'] == 'blue']

        red_acft_killed = red_group['acft_killed'].max() if not red_group.empty else 0
        red_acft_standing = red_group['acft_standing'].min() if not red_group.empty else 0
        red_bmb_released = red_group['bmb_released'].max() if not red_group.empty else 0
        red_wind_S_ou_N = red_group['wind_S_ou_N'].max() if not red_group.empty else 0
        red_wind_ang = red_group['wind_ang'].max() if not red_group.empty else 0
        red_wind_speed_factor = red_group['wind_speed_factor'].max() if not red_group.empty else 0
        red_wind_tempo_reta = red_group['wind_tempo_reta'].max() if not red_group.empty else 0
        red_WEZ_ratio = red_group['WEZ_ratio'].max() if not red_group.empty else 0
            
        blue_building_damage = blue_group['building_damage'].max() if not blue_group.empty else 0
        blue_sam_hit = blue_group['sam_hit'].max() if not blue_group.empty else 0
        blue_sam_lost = blue_group['sam_lost'].max() if not blue_group.empty else 0
        blue_sam_remaining = blue_group['sam_remaining'].min() if not blue_group.empty else 0
        blue_sim_time = blue_group['sim_time'].max() if not blue_group.empty else "00:00.00"

        summary_data.append({
            'exp': exp,
            'red_acft_killed': red_acft_killed,
            'red_acft_standing': red_acft_standing,
            'red_bmb_released': red_bmb_released,
            'building_damage': blue_building_damage,
            'blue_sam_hit': blue_sam_hit,
            'blue_sam_lost': blue_sam_lost,
            'blue_sam_remaining': blue_sam_remaining,
            'sim_time': blue_sim_time,        
            'wind_S_ou_N': red_wind_S_ou_N,
            'wind_ang': red_wind_ang,
            'wind_speed_factor': red_wind_speed_factor,
            'wind_tempo_reta': red_wind_tempo_reta, 
            'WEZ_ratio': red_WEZ_ratio           
        })

    summary_df = pd.DataFrame(summary_data)
    return summary_df


# In[82]:


summary_df = create_summary(df_clean_2)
summary_df.head()


# ### Transformando o tempo para segundos

# In[83]:


def convert_time_to_seconds(time_str):
    minutes, seconds = time_str.split(':')
    return int(minutes) * 60 + float(seconds)


# In[84]:


summary_df['sim_time'] = summary_df['sim_time'].apply(convert_time_to_seconds)
summary_df.head()


# ### Incorporando ao data frame a respectiva seed de cada experimento

# In[90]:


summary_df = summary_df.assign(seed=df['seed'])
summary_df.head()


# ## Armazenando as informações em **csv** para analise no _Rhead

# In[88]:


simulation_name = "resultado_refinado_Winding_F_A"
file_name = f'{simulation_name}.csv'
merged_df.to_csv(file_name, index=False)


# In[ ]:




