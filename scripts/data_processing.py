import pandas as pd
import numpy as np



def main():
    df_t = pd.read_csv("data/2t.csv")
    df_p = pd.read_csv("data/tp.csv")
    df_yield = pd.read_csv("data/yield.csv")

    df_t.rename(columns={"state_name": "state", "2m max temperature": "temp_K"}, inplace=True)
    df_p.rename(columns={"state_name": "state", "total precipitation": "precip_mm"}, inplace=True)
    
    df_t.iloc[:, 2:] = df_t.iloc[:, 2:] - 273.15
    df_t.rename(columns={"temp_K": "temp_C"}, inplace=True)
    
    # reshape date
    df_t["date"] = pd.to_datetime(df_t["date"])
    df_t["year"] = df_t["date"].dt.year
    df_t["month"] = df_t["date"].dt.month

    df_p["date"] = pd.to_datetime(df_p["date"])
    df_p["year"] = df_p["date"].dt.year
    df_p["month"] = df_p["date"].dt.month

    #filter by growing season (ie. May–September)
    grow_months = [5, 6, 7, 8, 9]
    df_t_grow = df_t[df_t["month"].isin(grow_months)]
    df_p_grow = df_p[df_p["month"].isin(grow_months)]

    # Merge dataframes
    t_features = df_t_grow.groupby(["state", "year"]).agg(
        avg_temp_C=("temp_C", "mean"),
        max_temp_C=("temp_C", "max"),
        min_temp_C=("temp_C", "min"),
    ).reset_index()

    p_features = df_p_grow.groupby(["state", "year"]).agg(
        total_precip_mm=("precip_mm", "sum"),
        avg_precip_mm=("precip_mm", "mean")
    ).reset_index()

    df_yield.rename(columns={"state_name": "state"}, inplace=True)  
    full_df = df_yield.merge(t_features, on=["state", "year"]).merge(p_features, on=["state", "year"])
    
    # Hot days > 35°C
    hot_days = df_t_grow[df_t_grow["temp_C"] > 35] \
        .groupby(["state", "year"]).size().reset_index(name="hot_days")

    # Freezing days < 5°C
    freezing_days = df_t_grow[df_t_grow["temp_C"] < 5] \
        .groupby(["state", "year"]).size().reset_index(name="freezing_days")

    # Dry days < 1 mm
    dry_days = df_p_grow[df_p_grow["precip_mm"] < 1] \
        .groupby(["state", "year"]).size().reset_index(name="dry_days")

    # Flooding days > 20 mm
    flooding_days = df_p_grow[df_p_grow["precip_mm"] > 20] \
        .groupby(["state", "year"]).size().reset_index(name="flooding_days")
        
    extreme_features = hot_days \
        .merge(freezing_days, on=["state", "year"], how="outer") \
        .merge(dry_days, on=["state", "year"], how="outer") \
        .merge(flooding_days, on=["state", "year"], how="outer")

    # Fill NaNs with 0 (no extreme events observed in that year)
    extreme_features.fillna(0, inplace=True)

    # Then merge with your main feature set
    full_df2 = full_df.merge(extreme_features, on=["state", "year"], how="left")
    
    full_df2.drop(columns=["freezing_days"], inplace=True)


    full_df2.to_csv("data/df_processed.csv", index=False)
    print("✅ Processed data saved to data/df_processed.csv")

if __name__ == "__main__":
    main()