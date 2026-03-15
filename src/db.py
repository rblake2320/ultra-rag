from psycopg2 import connect
from psycopg2.extras import DictCursor

def get_conn():
    # Connect to the database
    conn = connect(
        host="localhost",
        database="ultra_rag",
        user="ultra_rag",
        password="ultra_rag",
        cursor_factory=DictCursor
    )
    return conn
