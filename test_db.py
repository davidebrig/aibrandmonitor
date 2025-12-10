from sqlalchemy import create_engine
import pandas as pd

URL = "postgresql+psycopg2://postgres:i36wBZ0MoP8JfR4v@db.gjnpcpsvlbnrdhyfputd.supabase.co:5432/postgres?sslmode=require"

engine = create_engine(URL)

try:
    df = pd.read_sql("SELECT now()", engine)
    print("CONNESSIONE OK!")
    print(df)
except Exception as e:
    print("ERRORE DI CONNESSIONE:")
    print(e)