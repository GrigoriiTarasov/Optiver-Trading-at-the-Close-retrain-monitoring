from itertools import combinations
import pandas as pd
import polars as pl
import reduce_mem_df

def stock_ind_weight(df, weights):
    weight_df = pd.DataFrame()
    weight_df['stock_id'] = list(range(200))
    weight_df['weight'] = weights
     
    df = df.merge(weight_df,
                  how='left',
                  on=['stock_id'])
    return df
def scale_size_cols(df,
                    scale_dict,
                    size_col = ['imbalance_size','matched_size','bid_size','ask_size']):
    for _ in size_col:
        tmp_map = scale_dict[_].copy()
        tmp_map = {int(k):v 
                       for k,v in tmp_map.items()}
        df[f"{_}_stock_median"] = df['stock_id'].map(tmp_map)
        df[f"scale_{_}"] = df[_] / df[f"{_}_stock_median"]
        del df[f"{_}_stock_median"]

    return df
def imbalance_flag(df):
    #buy-side imbalance; 1
    #sell-side imbalance; -1
    #no imbalance; 0
    df['auc_bid_size'] = df['matched_size']
    df['auc_ask_size'] = df['matched_size']
    df.loc[df['imbalance_buy_sell_flag']==1,'auc_bid_size'] += df.loc[df['imbalance_buy_sell_flag']==1,
                                                                      'imbalance_size']
    df.loc[df['imbalance_buy_sell_flag']==-1,'auc_ask_size'] += df.loc[df['imbalance_buy_sell_flag']==-1,
                                                                      'imbalance_size']
    return df
def feats_orderbook(df):
    df = df.with_columns([
        (pl.col('ask_size') * pl.col('ask_price')).alias("ask_money"),
        (pl.col('bid_size') * pl.col('bid_price')).alias("bid_money"),
        (pl.col('ask_size') + pl.col("auc_ask_size")).alias("ask_size_all"),
        (pl.col('bid_size') + pl.col("auc_bid_size")).alias("bid_size_all"),
        (pl.col('ask_size') + pl.col("auc_ask_size") + pl.col('bid_size') + pl.col("auc_bid_size")).alias("volumn_size_all"),
        (pl.col('reference_price') * pl.col('auc_ask_size')).alias("ask_auc_money"),
        (pl.col('reference_price') * pl.col('auc_bid_size')).alias("bid_auc_money"),
        (pl.col('ask_size') * pl.col('ask_price') + pl.col('bid_size') * pl.col('bid_price')).alias("volumn_money"),
        (pl.col('ask_size') + pl.col('bid_size')).alias('volume_cont'),
        (pl.col('ask_size') - pl.col('bid_size')).alias('diff_ask_bid_size'),
        (pl.col('imbalance_size') + 2 * pl.col('matched_size')).alias('volumn_auc'),
        ((pl.col('imbalance_size') + 2 * pl.col('matched_size')) * pl.col("reference_price")).alias('volumn_auc_money'),
        ((pl.col('ask_price') + pl.col('bid_price'))/2).alias('mid_price'),
        ((pl.col('near_price') + pl.col('far_price'))/2).alias('mid_price_near_far'),
        (pl.col('ask_price') - pl.col('bid_price')).alias('price_diff_ask_bid'),
        (pl.col('ask_price') / pl.col('bid_price')).alias('price_div_ask_bid'),
        (pl.col('imbalance_buy_sell_flag') * pl.col('scale_imbalance_size')).alias('flag_scale_imbalance_size'),
        (pl.col('imbalance_buy_sell_flag') * pl.col('imbalance_size')).alias('flag_imbalance_size'),
        (pl.col('imbalance_size') / pl.col('matched_size') * pl.col('imbalance_buy_sell_flag')).alias("div_flag_imbalance_size_2_balance"),
        ((pl.col('ask_price') - pl.col('bid_price')) * pl.col('imbalance_size')).alias('price_pressure'),
        ((pl.col('ask_price') - pl.col('bid_price')) * pl.col('imbalance_size') * pl.col('imbalance_buy_sell_flag')).alias('price_pressure_v2'),
        ((pl.col("ask_size") - pl.col("bid_size")) / (pl.col("far_price") - pl.col("near_price"))).alias("depth_pressure"),
        (pl.col("bid_size") / pl.col("ask_size")).alias("div_bid_size_ask_size"),
    ])
    return df
def scalar_divs(df, feas_list):
    
    add_cols = []
    for col1, col2 in [
        ("imbalance_size","bid_size"),
        ("imbalance_size","ask_size"),
        ("matched_size","bid_size"),
        ("matched_size","ask_size"),
        ("imbalance_size","volume_cont"),
        ("matched_size","volume_cont"),
        ("auc_bid_size","bid_size"),
        ("auc_ask_size","ask_size"),
        ("bid_auc_money","bid_money"),
        ("ask_auc_money","ask_money"),
    ]:
        add_cols.append((pl.col(col1) / pl.col(col2)).alias(f"div_{col1}_2_{col2}"))
        feas_list.append(f"div_{col1}_2_{col2}")
    return df.with_columns(add_cols)
    
def imbalances_volumes(df, feas_list):
    add_cols = []
    for pair1,pair2 in [
        ('ask_size','bid_size'),
        ('ask_money','bid_money'),
        ('volumn_money','volumn_auc_money'),
        ('volume_cont','volumn_auc'),
        ('imbalance_size','matched_size'),
        ('auc_ask_size','auc_bid_size'),
        ("ask_size_all",'bid_size_all')
    ]:
        col_imb = f"imb1_{pair1}_{pair2}"
        add_cols.extend([
            ((pl.col(pair1) - pl.col(pair2)) / (pl.col(pair1) + pl.col(pair2))).alias(col_imb),
        ])
        feas_list.extend([col_imb])
    return df.with_columns(add_cols)
def imbalances_prices(df,feas_list):
    
    fea_append_list = []
    prices = ["reference_price", "far_price", "near_price", "ask_price", "bid_price", "wap","mid_price"]
    for c in combinations(prices, 2):
        fea_append_list.append(((pl.col(c[0]) - pl.col(c[1])) / (pl.col(c[0]) + pl.col(c[1]))).alias(f"imb1_{c[0]}_{c[1]}"))
        feas_list.extend([f"imb1_{c[0]}_{c[1]}"])

    return df.with_columns(fea_append_list)
def urgencies(df):
    df = df.with_columns([
        ((pl.col("imb1_ask_size_bid_size") + 2) * (pl.col("imb1_ask_price_bid_price") + 2) * (pl.col("imb1_auc_ask_size_auc_bid_size")+2)).alias("market_urgency_v2"),
        (pl.col('price_diff_ask_bid') * (pl.col('imb1_ask_size_bid_size'))).alias('market_urgency'),
        (pl.col('imb1_ask_price_bid_price') * (pl.col('imb1_ask_size_bid_size'))).alias('market_urgency_v3'),
    ])
    return df
def stat_of_significant(df):
    
    add_cols = []
    for col in ["bid_auc_money","imb1_reference_price_wap","bid_size_all",
                "imb1_auc_ask_size_auc_bid_size","div_flag_imbalance_size_2_balance",
                "imb1_ask_size_all_bid_size_all","flag_imbalance_size","imb1_reference_price_mid_price"]:
        for window in [3,6,18,36,60]:
            add_cols.append(pl.col(col).rolling_mean(window_size=window,min_periods=1).over('stock_id','date_id').alias(f'rolling{window}_mean_{col}'))
            add_cols.append(pl.col(col).rolling_std(window_size=window,min_periods=1).over('stock_id','date_id').alias(f'rolling{window}_std_{col}'))
  
            #feas_list.extend([f'rolling{window}_mean_{col}',f'rolling{window}_std_{col}'])
    return df.with_columns(add_cols)
def imb_spread_momentum(df,feas_list):
        
    df = df.with_columns([
        pl.col("flag_imbalance_size").diff().over('stock_id','date_id').alias("imbalance_momentum_unscaled"),
        pl.col("price_diff_ask_bid").diff().over('stock_id','date_id').alias("spread_intensity"),
    ])
    feas_list.extend(["imbalance_momentum_unscaled","spread_intensity"])
    df = df.with_columns([
        (pl.col("imbalance_momentum_unscaled")/pl.col("matched_size")).alias("imbalance_momentum")
    ])
    feas_list.extend(["imbalance_momentum"])
    return df
def univar_diff(df, feas_list):
    #Calculate diff features for specific columns
    add_cols = []
    for col in ['ask_price',
 'bid_price',
 'imb1_reference_price_near_price',
 'bid_size',
 'scale_bid_size',
 'mid_price',
 'ask_size',
 'price_div_ask_bid',
 'div_bid_size_ask_size',
 'market_urgency',
 'wap',
 'imbalance_momentum']:
        for window in [1, 2, 3, 10]:
            add_cols.append((pl.col(col).diff(window).over('stock_id','date_id')).alias(f"{col}_diff_{window}"))
            feas_list.append(f"{col}_diff_{window}")
    return df.with_columns(add_cols)
def target_shift(df):
    for shift_period in [1,3,12,6]:
    
        df = df.with_columns([
            pl.col("wap").shift(-shift_period).over("stock_id","date_id").alias(f"wap_shift_n{shift_period}")
        ])
        df = df.with_columns([
            (pl.col(f"wap_shift_n{shift_period}")/pl.col("wap")).alias("target_single")
        ])

        df = df.with_columns([
            pl.when(pl.col("target_single").is_null())
            .then(0)
            .otherwise(pl.col("weight"))
            .alias("weight_tmp")
        ])

        df = df.with_columns([
            (((pl.col("weight_tmp") * pl.col("target_single")).sum().over("date_id","seconds_in_bucket")) / ((pl.col("weight_tmp")).sum().over("date_id","seconds_in_bucket"))).alias("index_target_mock")
        ])

        df = df.with_columns([
            ((pl.col("target_single") - pl.col("index_target_mock"))*10000).alias("target_mock")
        ])

        df = df.with_columns([
            pl.col("target_mock").shift(shift_period).over("stock_id","date_id").alias(f"target_mock_shift{shift_period}"),
        ])
        print(f"{shift_period} is done")
    return df
def stat_target(df):
    add_cols = []
    for col in ['target_mock_shift6','target_mock_shift1','target_mock_shift3','target_mock_shift12']:
        for window in [1, 3, 6, 12, 24, 48]:
            add_cols.append(pl.col(col).rolling_mean(window_size=window,min_periods=1).over('stock_id','date_id').alias(f'rolling{window}_mean_{col}'))
    return df.with_columns(add_cols)
def significants_base_transforms(df):
    add_cols = []
    for col in ["imb1_auc_ask_size_auc_bid_size","flag_imbalance_size","price_pressure_v2","scale_matched_size"]:
        for window_size in [1,2,3,6,12]:
            add_cols.append(pl.col(col).shift(window_size).over('stock_id','date_id').alias(f'shift{window_size}_{col}'))
            add_cols.append((pl.col(col) / pl.col(col).shift(window_size).over('stock_id','date_id')).alias(f'div_shift{window_size}_{col}'))
            add_cols.append((pl.col(col) - pl.col(col).shift(window_size).over('stock_id','date_id')).alias(f'diff_shift{window_size}_{col}'))
    
    df = df.with_columns(add_cols)
    return df
def ind_weighted_significants2(df,feas_list):
    add_cols = []
    for col in ['imb1_ask_price_mid_price',
                 'market_urgency',
                 'market_urgency_diff_1',
                 'imb1_ask_money_bid_money',
                 'rolling18_mean_imb1_ask_size_all_bid_size_all',
                 'rolling18_mean_imb1_auc_ask_size_auc_bid_size',
                 'rolling18_mean_imb1_reference_price_wap',
                 'ask_price_diff_3',
                 'diff_shift1_price_pressure_v2',
                 'diff_shift12_scale_matched_size',
                 'diff_shift1_flag_imbalance_size',
                 'imb1_ask_size_bid_size',
                 'imb1_bid_price_mid_price',
                 'rolling48_mean_target_mock_shift6']:
        
        add_cols.append(
            (((pl.col(col) * pl.col("weight")).sum().over("date_id","seconds_in_bucket"))/\
            (((pl.col("weight")).sum().over("date_id","seconds_in_bucket")))).alias(f"global_{col}"))
        
        feas_list.append(f"global_{col}")
        
    return df.with_columns(add_cols)
#from memory_profiler import profile
#@profile
def generate_features_no_hist_polars(df,
                                     index_stock_weights,
                                     scale_dict):
    '''Arguments depends on split'''
    #print(-1)
    #reduce_mem_df.numeric(df)
    #print(0)
    df = stock_ind_weight(df, index_stock_weights)
    df = scale_size_cols(df, scale_dict)
    
    df = imbalance_flag(df)    
    print(1)
    df = pl.from_pandas(df)
    
    
    print(2)
    feas_list = ['stock_id','seconds_in_bucket','imbalance_size','imbalance_buy_sell_flag',
               'reference_price','matched_size','far_price','near_price','bid_price','bid_size',
                'ask_price','ask_size','wap','scale_imbalance_size','scale_matched_size','scale_bid_size','scale_ask_size'
                 ,'auc_bid_size','auc_ask_size']
    df = feats_orderbook(df)
    feas_list.extend(['ask_money', 'bid_money', 'ask_auc_money','bid_auc_money',"ask_size_all","bid_size_all","volumn_size_all",
                      'volumn_money','volume_cont',"volumn_auc","volumn_auc_money","mid_price",
                      'mid_price_near_far','price_diff_ask_bid',"price_div_ask_bid","flag_imbalance_size","div_flag_imbalance_size_2_balance",
                     "price_pressure","price_pressure_v2","depth_pressure","flag_scale_imbalance_size","diff_ask_bid_size"])        
    df = scalar_divs(df, feas_list)
    df = imbalances_volumes(df, feas_list)
    df = imbalances_prices(df,feas_list)
    df = urgencies(df)

    print(3)
    
    feas_list = ['imb1_wap_mid_price', 'imb1_ask_money_bid_money', 'imb1_volume_cont_volumn_auc', 'imb1_reference_price_ask_price', 
                 'imb1_reference_price_mid_price', 'seconds_in_bucket', 'div_flag_imbalance_size_2_balance', 'ask_price', 
                 'imb1_reference_price_bid_price', 'scale_matched_size', 'imb1_near_price_wap', 'volumn_auc_money', 'imb1_far_price_wap', 
                 'bid_size', 'scale_bid_size', 'bid_size_all']
    df = stat_of_significant(df)
    print(4)
    
    feas_list = ['imb1_wap_mid_price', 'imb1_ask_money_bid_money', 'imb1_volume_cont_volumn_auc', 
                     'imb1_reference_price_ask_price', 'imb1_reference_price_mid_price', 
                     'seconds_in_bucket', 'div_flag_imbalance_size_2_balance', 'ask_price', 
                     'imb1_reference_price_bid_price', 'scale_matched_size', 'imb1_near_price_wap', 
                     'volumn_auc_money', 'imb1_far_price_wap', 'bid_size', 'scale_bid_size', 'bid_size_all', 
                     'rolling18_mean_imb1_auc_ask_size_auc_bid_size', 'rolling3_mean_div_flag_imbalance_size_2_balance', 
                     'rolling60_std_div_flag_imbalance_size_2_balance', 'rolling36_mean_flag_imbalance_size', 
                     'rolling3_std_imb1_auc_ask_size_auc_bid_size', 'rolling18_mean_imb1_ask_size_all_bid_size_all', 
                     'rolling6_mean_div_flag_imbalance_size_2_balance', 'rolling6_std_imb1_auc_ask_size_auc_bid_size', 
                     'rolling3_mean_imb1_auc_ask_size_auc_bid_size', 'rolling60_std_imb1_auc_ask_size_auc_bid_size', 
                     'rolling6_std_bid_size_all', 'rolling3_std_bid_size_all', 'rolling3_mean_bid_size_all', 
                     'rolling18_std_bid_auc_money', 'rolling36_mean_bid_auc_money',"rolling60_mean_imb1_reference_price_wap",
                    'rolling18_mean_imb1_reference_price_wap', 'rolling3_mean_imb1_reference_price_mid_price']
    df = imb_spread_momentum(df,feas_list)
    print(5)
    df = univar_diff(df, feas_list)
    print(6)
    df = target_shift(df)
    print(7)
    df = stat_target(df) 
    print(8)
    
    keep_cols_new = ['rolling48_mean_target_mock_shift3', 'rolling48_mean_target_mock_shift1', 'rolling48_mean_target_mock_shift12',
'rolling1_mean_target_mock_shift6', 'rolling24_mean_target_mock_shift6','rolling24_mean_target_mock_shift12',]
    feas_list.extend(keep_cols_new)
    df = significants_base_transforms(df)
    feas_list.extend(['div_shift6_imb1_auc_ask_size_auc_bid_size',
                     'diff_shift6_price_pressure_v2',
                     'shift1_price_pressure_v2',
                     'div_shift3_flag_imbalance_size',
                     'div_shift12_imb1_auc_ask_size_auc_bid_size',
                     'div_shift3_scale_matched_size',
                     'diff_shift6_flag_imbalance_size',
                     'shift12_imb1_auc_ask_size_auc_bid_size',
                     'div_shift12_price_pressure_v2',
                     'shift6_flag_imbalance_size',
                     'diff_shift3_imb1_auc_ask_size_auc_bid_size',
                     'div_shift12_flag_imbalance_size',
                     'shift12_flag_imbalance_size'])

    df = ind_weighted_significants2(df,feas_list)
    # MACD
    rsi_cols = ["mid_price_near_far","imb1_reference_price_wap","near_price",]
    add_cols = []
    for col in rsi_cols:
        for window_size in [3,6,12,24,48]:
            add_cols.append(pl.col(col).ewm_mean(span=window_size, adjust=False).over('stock_id','date_id').alias(f"rolling_ewm_{window_size}_{col}"))
            #feas_list.append(f"rolling_ewm_{window_size}_{col}")
    df = df.with_columns(add_cols)
    
    add_cols = []
    for col in rsi_cols:
        for w1,w2 in zip((3,6,12,24),(6,12,24,48)):
            add_cols.append((pl.col(f"rolling_ewm_{w1}_{col}") - pl.col(f"rolling_ewm_{w2}_{col}")).alias(f"dif_{col}_{w1}_{w2}"))
            #feas_list.append(f"dif_{col}_{w1}_{w2}")
    df = df.with_columns(add_cols)
    
    add_cols = []
    for col in rsi_cols:
        for w1,w2 in zip((3,6,12,24),(6,12,24,48)):
            add_cols.append(pl.col(f"dif_{col}_{w1}_{w2}").ewm_mean(span=9, adjust=False).over('stock_id','date_id').alias(f"dea_{col}_{w1}_{w2}"))
            #feas_list.append(f"dea_{col}_{w1}_{w2}")
    df = df.with_columns(add_cols)
    
    add_cols = []
    for col in rsi_cols:
        for w1,w2 in zip((3,6,12,24),(6,12,24,48)):
            add_cols.append((pl.col(f"dif_{col}_{w1}_{w2}") - pl.col(f"dea_{col}_{w1}_{w2}")).alias(f"macd_{col}_{w1}_{w2}"))
            #feas_list.append(f"macd_{col}_{w1}_{w2}")
    
    feas_list.extend(['macd_imb1_reference_price_wap_12_24',
 'dif_imb1_reference_price_wap_3_6',
 'macd_mid_price_near_far_12_24',
 'dif_near_price_3_6',
 'macd_near_price_24_48',
 'dea_imb1_reference_price_wap_12_24',
 'macd_near_price_12_24',
 'rolling_ewm_24_imb1_reference_price_wap',
 'dif_near_price_6_12',
 'dea_mid_price_near_far_6_12',
 'dea_near_price_24_48',
 'rolling_ewm_12_imb1_reference_price_wap',
 'dif_imb1_reference_price_wap_12_24'])
    df = df.with_columns(add_cols)
    
    add_cols = []
    for col in ["target"]:
        # 176 1,2,3,5,10,15,20,25,30
        # [1,2,3,5,10,15,20,25,30,35,40,45,60] 5.8704926 157
        # [1,2,3,5,10,15,20,30,45,60] 5.8708683137
        for window_size in [1,2,3,5,10,15,20,25,30,35,40,45,60]:
            add_cols.append(pl.col(col).shift(1).rolling_mean(window_size=window_size,min_periods=1).over('stock_id','seconds_in_bucket').alias(f'rolling_mean_{window_size}_{col}_second'))
            add_cols.append(pl.col(col).shift(1).rolling_std(window_size=window_size,min_periods=1).over('stock_id','seconds_in_bucket').alias(f'rolling_std_{window_size}_{col}_second'))

            
            feas_list.extend([f'rolling_mean_{window_size}_{col}_second',f'rolling_std_{window_size}_{col}_second',])

    df = df.with_columns(add_cols)
    
    
    return df, feas_list
        
if  __name__ == "__main__":
    import json
    import pickle
    
    settings_path = '../configs/settings.json'
    with open(settings_path, 'r') as f:
        config = json.load(f)
    
    # EDA 3)
    with open(f"../{config['RAW_DATA_DIR']}/index_stock_weights.pkl", 
              'rb') as f:
        index_stock_weights = pickle.load(f)   

    with open(f"../{config['RAW_DATA_DIR']}/scale_dict.pkl", 'rb') as f:
        scale_dict = pickle.load(f)

    df_raw = pd.read_csv(f"../{config['RAW_DATA_DIR']}/train.csv")

    df, feas_list = generate_features_no_hist_polars(df_raw,
                                          index_stock_weights,
                                          scale_dict)
    print(f'{df.shape=}')

    