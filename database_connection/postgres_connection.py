import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dataclasses import dataclass
from dotenv import load_dotenv
from constant import POSTGRES_CRED

#------- Important-------------
# If the database needs to be changed or a new database needs to be added please consider,
#  this class template to make new database clients 

load_dotenv()

# This is used to create a default class that will
# get the connection cred form the .env file 
@dataclass
class postgres_config:
    dbname=os.getenv(POSTGRES_CRED.DB_NAME)
    user=os.getenv(POSTGRES_CRED.DB_USER)
    password=os.getenv(POSTGRES_CRED.DB_PASSWORD)
    host=os.getenv(POSTGRES_CRED.DB_HOST)
    port=os.getenv(POSTGRES_CRED.DB_PORT)

# Instance of default connection
default_config = postgres_config()

# A Reusable postgres connection client 
class PostgresClient:
    def __init__(self,config=default_config):
        self.connection = None
        self.cursor = None
        self.config = config

    def connect(self):
        """Connect to the PostgreSQL database."""
        if self.connection is None:
            self.connection = psycopg2.connect(
                dbname=self.config.dbname,
                user=self.config.user,
                password=self.config.password,
                host=self.config.host,
                port=self.config.port,
            )
            # initilizing the cursor object
            self.cursor = self.connection.cursor(cursor_factory=RealDictCursor)

    # for executing the query  
    def execute_query(self, 
                      query, 
                      params=None, 
                      fetch_one=False, 
                      fetch_all=False):
        """
        Execute a SQL query.

        :param query: SQL query string
        :param params: Optional tuple/dict of query parameters
        :param fetch_one: Return a single row
        :param fetch_all: Return all rows
        """
        if self.connection is None or self.cursor is None:
            raise RuntimeError("Database connection is not established. Call connect() first.")

        self.cursor.execute(query, params)
        self.connection.commit()

        if fetch_one:
            return self.cursor.fetchone()
        if fetch_all:
            return self.cursor.fetchall()

        return None

    # close the existing cursor and connection object 
    def close(self):
        """Close cursor and database connection."""
        if self.cursor:
            self.cursor.close()
            self.cursor = None

        if self.connection:
            self.connection.close()
            self.connection = None