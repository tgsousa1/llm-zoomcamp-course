import pandas as pd
import json
import re

from pathlib import Path

RAW_DATA_PATH = Path("data/raw")
PROCESSED_DATA_PATH = Path("data/processed")

OUTPUT_FILE = PROCESSED_DATA_PATH / "documents.json"

def load_movies(data_path):
    dataframes=[] #empty list of dataframes

    csv_files=data_path.glob("*.csv")

    for file in csv_files:
        df=pd.read_csv(file)
        dataframes.append(df)

    final_data=pd.concat(dataframes, ignore_index=True) #concat all dfs in list

    return final_data

#df=load_movies(RAW_DATA_PATH)
#print(df.shape)
#print(df["title"].head())

def clean_text(txt):
    if pd.isna(txt):
        return ""
    
    txt = re.sub(r"\[\d+\]", "", txt)   # remove [1], [25], etc
    txt = re.sub(r"\s+", " ", txt)      # remove double spaces

    return txt.strip()

def prepare_document(df):
    documents= []

    for idx,(_,row) in enumerate(df.iterrows()):
        content=f"""
Title: {clean_text(row["title"])}

Description: {clean_text(row["description"])}

Directed by: {clean_text(row["directed_by"])}

Written by: {clean_text(row["written_by"])}

Produced by: {clean_text(row["produced_by"])}

Starring: {clean_text(row["starring"])}

Release date: {clean_text(row["release_date"])}

Country: {clean_text(row["country"])}

Language: {clean_text(row["language"])}
    """.strip()

        documents.append({
            "id":idx,
            "title":clean_text(row["title"]),
            "content":content
        })

    return documents

def save_docs(documents, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)

def main():
    df = load_movies(RAW_DATA_PATH)
    documents = prepare_document(df)
    save_docs(documents, OUTPUT_FILE)

    print(f"Saved {len(documents)} documents to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()