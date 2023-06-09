########################################################################
# ASSOCATING RULES PROJECT WITH online_retail_2 DATA
########################################################################

##################################
# Preparing Data
##################################
# !pip install mlxtend
import pandas as pd
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
import datetime as dt
# çıktının tek bir satırda olmasını sağlar.
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules

df_ = pd.read_excel("datasets/online_retail_II.xlsx",
                    sheet_name="Year 2010-2011")

df = df_.copy()

df.head()
df.info()
df.isnull().sum()


df = df.dropna()

df = df[df["StockCode"] != "POST"]

# If Invoice contains C, it means it has been canceled
df = df[~df["Invoice"].str.contains("C",na=False)]

df = df[df["Price"] > 0]
df = df[df["Quantity"] > 0]

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

    def replace_with_thresholds(dataframe, variable):
        low_limit, up_limit = outlier_thresholds(dataframe, variable)
        # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
        dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit



replace_with_thresholds(df, "Price")
replace_with_thresholds(df, "Quantity")

######################
# Preparing Data for ARL
######################

df_de = df[df["Country"] == "Germany"]

def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(["Invoice", "StockCode"])["Quantity"].sum().unstack().fillna(0).applymap(lambda x:1 if x>0 else 0)
    else:
        return dataframe.groupby(["Invoice", "Description"])["Quantity"].sum().unstack().fillna(0).applymap(lambda x:1 if x>0 else 0)

fr_inv_pro_df = create_invoice_product_df(df_de, id=True)

def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)

frequent_itemsets = apriori(fr_inv_pro_df,
                            min_support=0.01,
                            use_colnames=True)

frequent_itemsets.sort_values("support",ascending=False)

rules = association_rules(frequent_itemsets,
                          metric="support",
                          min_threshold=0.01)

sorted_rules = rules.sort_values("lift", ascending=False)

def create_rules(dataframe):
    frequent_itemsets = apriori(dataframe, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets,
                              metric="support",
                              min_threshold=0.01,)
    sorted_rules = rules.sort_values("lift",ascending=False)

############################
# Creating arl_rules function
############################

product_id = 22629
recommendation_list = []

for i, product in enumerate(sorted_rules["antecedents"]):
    for j in list(product):
        if j == product_id:
            recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])


def arl_recommender(sorted_dataframe, product_id, rec_count=1):
    recommendation_list = []
    for i, product in enumerate(sorted_dataframe["antecedents"]):
        for j in list(product):
            if j == product_id:
             recommendation_list.append(list(sorted_dataframe.iloc[i]["consequents"])[0])

    return recommendation_list[0:rec_count]


arl_recommender(sorted_rules, 22629, 7)
arl_recommender(sorted_rules, 22328, 3)
arl_recommender(sorted_rules, 22961, 2)