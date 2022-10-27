import pandas as pd



def test_detect_DC():
    series = pd.Series([2,5,3,7,9,10,8,30,6])
    df = detect_DC(series)
    print(df)
    assert df["TREND"].to_list() == [-1,1,-1,1,1,1,-1,1,-1]

test_detect_DC()