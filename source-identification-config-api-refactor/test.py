import psycopg2

# Update these with your actual DB credentials
conn_params = {
    'dbname': 'sims',
    'user': 'docker',
    'password': 'docker',
    'host': 'localhost',
    'port': 5434
}

try:
    # Connect to the PostgreSQL database
    conn = psycopg2.connect(**conn_params)
    cursor = conn.cursor()

    # Execute the query
    cursor.execute("SELECT * FROM company;")

    # Fetch all results
    rows = cursor.fetchall()

    # Print each row
    for row in rows:
        print(row)

    # Clean up
    cursor.close()
    conn.close()

except Exception as e:
    print("Error:", e)
