import pandas as pd
df = pd.read_csv(r'C:\Users\SkyMap\Desktop\b\cap_xa_dang_cot.csv')
df_unpivot = pd.melt(df, id_vars=['Year','Xã'], value_vars=['1. Rừng', '2. Thảm cỏ', '3. Nước', '4. Khác'])
df_unpivot.to_csv(r'C:\Users\SkyMap\Desktop\b\cap_xa_dang_hang.csv')