pip install mlxtend
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules
df_ = pd.read_excel(r"C:\Users\yildi\OneDrive\Masaüstü\datasets/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()

df.info()
df.head()

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe

df = retail_data_prep(df)

df_gr = df[df['Country'] == "Germany"]

df_gr.head()

def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)

gr_inv_pro_df = create_invoice_product_df(df_gr)

gr_inv_pro_df.head()

gr_inv_pro_df = create_invoice_product_df(df_gr, id=True)

gr_inv_pro_df.head()

def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)

##Birliktelik Kuralları##

frequent_itemsets = apriori(gr_inv_pro_df, min_support=0.01, use_colnames=True)
frequent_itemsets.sort_values("support", ascending=False).head(50)

rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
rules.sort_values("support", ascending=False).head()



check_id(df_gr, 21987)
#'PACK OF 6 SKULL PAPER CUPS'#

check_id(df_gr, 23235)
#'STORAGE TIN VINTAGE LEAF'#

check_id(df_gr, 22747)
#"POPPY'S PLAYHOUSE BATHROOM"#

##Urün Onerisi##

#Id:21987#

product_id = 21987

sorted_rules = rules.sort_values("lift", ascending=False)
recommendation_list = []

for i, product in enumerate(sorted_rules["antecedents"]):
    for j in list(product):
        if j == product_id:
            recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

recommendation_list[0:1]

check_id(df_gr, 21086)

#'PACK OF 6 SKULL PAPER CUPS'# Önerisi : 'SET/6 RED SPOTTY PAPER CUPS'

def arl_recommender(rules_df, product_id, rec_count=1):

    sorted_rules = rules_df.sort_values("lift", ascending=False)

    recommendation_list = []

    for i, product in sorted_rules["antecedents"].items():
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"]))

    recommendation_list = list({item for item_list in recommendation_list for item in item_list})

    return recommendation_list[:rec_count]

arl_recommender(rules, 23235, 1)
check_id(df_gr, 20750)

#'STORAGE TIN VINTAGE LEAF'# Önerisi: 'RED RETROSPOT MINI CASES'

arl_recommender(rules, 22747, 1)
check_id(df_gr, 22423)

#"POPPY'S PLAYHOUSE BATHROOM"# Önerisi: 'REGENCY CAKESTAND 3 TIER'

