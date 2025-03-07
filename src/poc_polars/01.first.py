# Polars (lazy mode)

# Dataset de exemplo:
# NYC Taxi and Limousine Commission (TLC)
# https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
# Baixe os meses/anos que desejar.
# Tamanho dos arquivos: ~ 50GB/mês (Ex.: yellow_tripdata_2024-01.parquet)
# Numa estimativa grosseira, os 16 anos de dados completos podem ocupar ~ 9.6 GB em Parquet.

import time
import pandas as pd
import polars as pl

# 2 meses de dados em 2024 (janeiro e fevereiro) são 100 MB e suficientes para testes iniciais
# valid columns: ["VendorID", "tpep_pickup_datetime", "tpep_dropoff_datetime", "passenger_count",
#   "trip_distance", "RatecodeID", "store_and_fwd_flag", "PULocationID", "DOLocationID", "payment_type",
#   "fare_amount", "extra", "mta_tax", "tip_amount", "tolls_amount", "improvement_surcharge",
#   "total_amount", "congestion_surcharge", "Airport_fee"]

start = time.perf_counter()
df = pl.scan_parquet("data/parquet/nyctlc/*.parquet")
result = (
    df.filter(pl.col("RatecodeID").is_in([2, 3]))
      .group_by(["RatecodeID", "PULocationID"])
      .agg(pl.col("fare_amount").sum())
      .collect()  # Executa a query otimizada em Parquet com 'Predicate Pushdown'' e 'Projection Pruning''
)
end = time.perf_counter()
print(f" Polars: {(end-start)/10:.4f} segundos. Resultado = \n{result}")

start = time.perf_counter()
# Em Pandas (apesar de mais simples é menos eficiente)
df = pd.read_parquet("data/parquet/nyctlc/")
result = df[df["RatecodeID"] == 2].groupby("PULocationID")["fare_amount"].sum()
end = time.perf_counter()
print(f" Pandas: {(end-start)/10:.4f} segundos. "
      f"{result.shape} - Resultado = \n{result}")

print(f"Compare os resultados do shape dos DataFrames no Polars e no Pandas. "
      f"Veja que o Polars tem uma abordagem mais regular em termos de dados Tabulares.")
