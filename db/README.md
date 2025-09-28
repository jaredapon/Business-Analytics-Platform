# Database Setup

This folder contains initialization scripts and documentation for the **PostgreSQL** databases used in the Capstone Project:  
**Data-Driven Product Bundling Strategy and Pricing Optimization Using Market Basket Analysis and Prescriptive Analytics**.

---

## 📂 Folder Structure
```
db/
├─ init/
│ └─ 01-create-dbs.sql # SQL script to create required databases
└─ README.md
```

## 🗄️ Databases

Two main databases are required for the system:

1. **booklatte**  
   - Application database for storing business/analytics data.  
   - Used by ETL pipelines, ML scripts, and Dash for analytics visualization.  

2. **keycloak**  
   - Database for Keycloak authentication and identity management.  
   - Stores realms, clients, roles, and users.

---

## ⚙️ How Initialization Works

The SQL scripts inside `init/` are automatically executed by the Postgres container during startup.  
- `01-create-dbs.sql` ensures both **booklatte** and **keycloak** databases exist.  
- Additional `.sql` files can be added to this folder for creating tables, schemas, or seeding initial data.  

---

## ▶️ Running

1. Make sure you have your `.env` file configured with Postgres credentials.
2. Start services:
   ```bash
   docker compose up -d postgres
   ```
   The init/ scripts will automatically run inside the container.

   Verify databases:
   ```bash
   docker exec -it <postgres_container_id> psql -U $POSTGRES_USER -l
   ```

📝 Notes
- Do not commit production credentials to this folder.
- Use .env for secrets and only keep templates or examples in git.

If you need to reset the database:
```bash
docker compose down -v postgres
docker compose up -d postgres
```
(this will wipe and re-run init scripts)
