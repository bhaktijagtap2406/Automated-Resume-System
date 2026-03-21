import mysql.connector
import pandas as pd

# Connect
conn = mysql.connector.connect(
    host     = "localhost",
    user     = "root",
    password = "Manasvi@2005",        # ← add your password if you have one
    database = "resume_screening"
)
cursor = conn.cursor()

# Load CSV - UPDATE THIS PATH
df = pd.read_csv(
    "data/processed/resumes_nlp.csv")

df["NLP_Resume"]   = df["NLP_Resume"].fillna("")
df["Clean_Resume"] = df["Clean_Resume"].fillna("")
df["Category"]     = df["Category"].fillna("")
df["Years_Exp"]    = df["Years_Exp"].fillna(0)
df["Email"]        = df["Email"].fillna("")
df["Word_Count"]   = df["Word_Count"].fillna(0)

print(f"CSV loaded: {len(df)} rows")  # ← check CSV loads

sql = """
    INSERT INTO candidates
        (category, clean_resume,
         years_exp, email, word_count)
    VALUES (%s,%s,%s,%s,%s)
"""

count = 0
try:
    for _, row in df.iterrows():
        cursor.execute(sql, (
            str(row["Category"]),
            str(row["NLP_Resume"])[:65000],
            int(row["Years_Exp"]),
            str(row["Email"]),
            int(row["Word_Count"])
        ))
        count += 1
    conn.commit()
    print(f"Inserted {count} candidates!")

except Exception as e:
    print(f"ERROR: {e}")
    conn.rollback()

finally:
    cursor.close()
    conn.close()
