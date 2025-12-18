import pandas as pd

def padding_for_year(df, bin=None):
    if df['Time'].dt.year.nunique() != 1:
        raise Exception("The dataframe is empty or contains several years !")
    # get year and bin
    if not bin:
        temp_ds = df.head(2)['Time']
        year = temp_ds.iloc[0].year
        bin = (temp_ds.iloc[1] - temp_ds.iloc[0]).total_seconds() // 60
    else:
        year = df.sample(1)['Time'].iloc[0].year
    # 生成一年的时间序列 (bin分钟间隔) | get one-year serie (interval of 15 munites)
    start = pd.Timestamp(f"{year}-01-01 00:00:00")
    end = pd.Timestamp(f"{year}-12-31 23:45:00") 
    full_index = pd.date_range(start=start, end=end, freq=f"{bin}min")
    # 开始填充 | padding fill in
    df = df.set_index("Time")           # 先把 Time 变成 index | convert "Time" to index
    df = df.reindex(full_index, fill_value=0)  # 没有数据的地方补 0 | fill with 0
    df.index.name = "Time"              # 恢复 index 名 | rename the index as Time
    df = df.reset_index()               # 如果需要 Time 列 | reconvert "Time" to a normal column 

    return df

def convert_for_EV(chargelogs, padding=True, bin=15):
    df = chargelogs[['Start', "End", "ChargePointLabel", 'MeterTotal', 'DurationTotal']].copy()
    # 计算功率 | calculate power (per second)
    epsilon = 1e-9
    df['power']= df['MeterTotal'] / ( df['DurationTotal'] / 3600 + epsilon )
    # 对齐到 bin 分钟 (alignement to bin minutes)
    df["bin_start"] = df["Start"].dt.floor(f"{bin}min")
    df["bin_end"] = df["End"].dt.ceil(f"{bin}min")

    # 生成覆盖的 bins | create the bins
    df["Time"] = df.apply(
        lambda r: pd.date_range(r["bin_start"], r["bin_end"], freq=f"{bin}min"),
        axis=1
    )

    df = df.explode("Time")

    # 计算 overlap | calculate the overlap
    df["Time_end"] = df["Time"] + pd.Timedelta(minutes=bin)

    df["overlap_start"] = df[["Start", "Time"]].max(axis=1)
    df["overlap_end"]   = df[["End", "Time_end"]].min(axis=1)

    df["seconds"] = (
        df["overlap_end"] - df["overlap_start"]
    ).dt.total_seconds().clip(lower=0)

    # df["consomation"] = df["seconds"] * df['power'] / 1000
    df['true_power'] = (df['power'] / 1000) * (df['seconds'] / (bin*60))

    # 聚合 | group by time and ChargePointLabel
    result = (
        df.groupby(["Time", "ChargePointLabel"])["true_power"]
        .sum()
        .unstack(fill_value=0)
    ).reset_index()

    # 去掉多余的 index 名 | remove the column name
    result.columns.name = None

    # 计算总充电量 | calculate the total sonsomation
    result['Total'] = result.drop(columns="Time").sum(axis=1)

    if padding:
        # 填充一年的时间 | fill with one-year
        return padding_for_year(result, bin=bin)
    return result


if __name__=="__main__":
    from pathlib import Path
    root_proj = Path("/Users/yk/Documents/Projects/Pre-PhD")
    year = "2023"
    chargelogs = pd.read_excel(root_proj / "DATA_SYSTEM_LIDL" / "Raw_chargelogs" / f"Chargelogs {year}.xlsx")
    # convert the data frame
    converted_df = convert_for_EV(chargelogs, bin=15)
    print(converted_df.head(20))
