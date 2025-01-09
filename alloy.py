import psycopg2
from psycopg2 import OperationalError

# Replace these with your AlloyDB credentials
ALLOYDB_HOST = "your-alloydb-instance-ip"
ALLOYDB_PORT = "5432"  # Default PostgreSQL port
ALLOYDB_USER = "your-username"
ALLOYDB_PASSWORD = "your-password"
ALLOYDB_DATABASE = "your-database"

def test_alloydb_connection():
    """Test connection to AlloyDB."""
    try:
        # Establish connection
        connection = psycopg2.connect(
            host=ALLOYDB_HOST,
            port=ALLOYDB_PORT,
            user=ALLOYDB_USER,
            password=ALLOYDB_PASSWORD,
            dbname=ALLOYDB_DATABASE
        )
        print("Connection to AlloyDB successful!")
        connection.close()  # Close connection after test
    except OperationalError as e:
        print(f"Failed to connect to AlloyDB: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    test_alloydb_connection()
