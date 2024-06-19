#!/bin/bash

# Default values

DEFAULT_DB_HOST="default_host"
DEFAULT_DB_PORT="default_port"
DEFAULT_PHRASE="default_phrase"
DEFAULT_DB_NAME="pipeline"
DEFAULT_DB_USER="postgres"
DEFAULT_DB_PASSWORD="pipe"
DEFAULT_SCHEMA="public"
DEFAULT_KEYWORD="keyword"
DEFAULT_PREFIX=""

# Command line arguments
DB_HOST=${1:-$DEFAULT_DB_HOST}
DB_PORT=${2:-$DEFAULT_DB_PORT}
PHRASE=${3:-$DEFAULT_PHRASE}
DB_NAME=${4:-$DEFAULT_DB_NAME}
DB_USER=${5:-$DEFAULT_DB_USER}
DB_PASSWORD=${6:-$DEFAULT_DB_PASSWORD}
SCHEMA=${7:-$DEFAULT_SCHEMA}
KEYPHRASE=${8:-$DEFAULT_KEYWORD}
PREFIX=${9:-$DEFAULT_PREFIX}

# Export the PGPASSWORD environment variable
export PGPASSWORD=$DB_PASSWORD

# Get the list of tables in the specified schema that contain the keyphrase
tables=$(psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -t -c "
SELECT tablename 
FROM pg_tables 
WHERE schemaname = '$SCHEMA' AND tablename LIKE '%$KEYPHRASE%';")

# Loop through the list of tables and rename each one
for table in $tables; do
    if [[ ! -z "$table" ]]; then
        new_table="${PREFIX}_${table}"
        echo "Renaming table: $table to $new_table"
        psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "ALTER TABLE $SCHEMA.$table RENAME TO $new_table;"
    fi
done

# Unset the PGPASSWORD environment variable
unset PGPASSWORD

echo "All tables in schema '$SCHEMA' containing keyphrase '$KEYPHRASE' have been renamed with prefix '$PREFIX'."