from utils import extract_data_from_sqlite
import yaml
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from transformers import pipeline
import sqlite3
import torch
import streamlit as st
from langchain_community.document_loaders import SQLDatabaseLoader
from langchain_community.utilities import SQLDatabase
from typing import List, Dict, Any

# Load config once
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

db_path = config['DB_FILE']
table_name = config['TABLE_NAME']
model_path = config['MODEL_PATH']  # Changed from hardcoded
max_chunk_size = config['MAX_CHUNK_SIZE']
overlap = config['OVERLAP']
similarity_threshold = config['SIMILARITY_THRESHOLD']


def sql_preprocessing(
    db_uri: str,
    table_name: str,
    content_columns: List[str],
    metadata_columns: List[str],
    chunk_size: int = 200,
    chunk_overlap: int = 50
) -> List[Dict[str, Any]]:

    def page_content_mapper(row: Dict[str, Any]) -> str:
        return ". ".join(str(row.get(col)) for col in content_columns if row.get(col)).strip()

    def metadata_mapper(row: Dict[str, Any]) -> Dict[str, Any]:
        return {col: row.get(col) for col in metadata_columns if row.get(col) is not None}

    try:
        db = SQLDatabase.from_uri(db_uri)
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return []

    query_columns = list(set(content_columns + metadata_columns + ['id']))
    query = f"SELECT {', '.join(query_columns)} FROM {table_name}"

    try:
        loader = SQLDatabaseLoader(
            query=query,
            db=db,
            page_content_mapper=page_content_mapper,
            metadata_mapper=metadata_mapper
        )
        docs = loader.load()
    except Exception as e:
        print(f"Error loading documents: {e}")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_documents(docs)

    return [
        {
            'id': doc.metadata.get('id'),
            'text_content': doc.page_content
        }
        for doc in chunks if 'id' in doc.metadata
    ]


def llm_pipeline(text: str) -> str:
    try:
        summarizer = pipeline("text2text-generation", model=model_path)
        result = summarizer(text)
        return result[0]['generated_text']
    except Exception as e:
        print(f"Summarization error: {e}")
        return ""


def update_summaries_in_db(db_file: str, table_name: str, record_id: str, summary_text: str):
    try:
        conn = sqlite3.connect(db_file.replace("sqlite:///", ""))
        cursor = conn.cursor()

        cursor.execute(f"""
            UPDATE {table_name}
            SET summary = ?
            WHERE id = ?;
        """, (summary_text, record_id))
        conn.commit()
    except sqlite3.Error as e:
        print(f"DB error for ID {record_id}: {e}")
    finally:
        conn.close()


def main():
    st.title("SQL Data Summarization")

    db_uri = f"sqlite:///{db_path}"
    raw_db_path = db_path.replace("sqlite:///", "")

    content_cols = ['text']
    metadata_cols = ['id']

    st.write(f"Processing data from **{db_path}** table **{table_name}**...")

    data = sql_preprocessing(
        db_uri=db_uri,
        table_name=table_name,
        content_columns=content_cols,
        metadata_columns=metadata_cols,
        chunk_size=max_chunk_size,
        chunk_overlap=overlap
    )

    if not data:
        st.error("No data found or error during preprocessing.")
        return

    st.write(f"Found {len(data)} entries to summarize.")
    progress = st.progress(0)

    for i, entry in enumerate(data):
        record_id = entry['id']
        content = entry['text_content'].strip()

        if content:
            st.write(f"Summarizing record ID: {record_id}")
            try:
                summary = llm_pipeline(content)
                st.write(f"Summary: {summary[:100]}...")
                update_summaries_in_db(raw_db_path, table_name, record_id, summary)
            except Exception as e:
                st.error(f"Error processing ID {record_id}: {e}")
        else:
            update_summaries_in_db(raw_db_path, table_name, record_id, "")
        progress.progress((i + 1) / len(data))

    if st.checkbox("Show all summaries"):
        conn = sqlite3.connect(raw_db_path)
        cursor = conn.cursor()
        try:
            results = cursor.execute(f"SELECT * FROM {table_name}").fetchall()
            for row in results:
                st.markdown(f"**{row[0]}**: {row[1]}")
        except sqlite3.Error as e:
            st.error(f"Failed to fetch summaries: {e}")
        finally:
            conn.close()

    st.success("Summarization and DB update complete!")


if __name__ == "__main__":
    main()
