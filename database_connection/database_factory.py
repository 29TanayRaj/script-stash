from database_connection.postgres_connection import PostgresClient
from constant import DATABASE

# factory class that retuns a connection of the database client
# autoconnect connects the 

class DatabaseConnectionFactory:
    def __new__(cls, connection_type: str, auto_connect: bool = True):

        if connection_type == DATABASE.POSTGRES:
            client = PostgresClient()
        else:
            raise ValueError(f"Unsupported connection type: {connection_type}")

        if auto_connect:
            client.connect()

        return client
    
