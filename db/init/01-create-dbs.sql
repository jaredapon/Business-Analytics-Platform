-- Create booklatte database if it does not exist
SELECT 'CREATE DATABASE booklatte'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'booklatte')\gexec

-- Create keycloak database if it does not exist
SELECT 'CREATE DATABASE keycloak'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'keycloak')\gexec