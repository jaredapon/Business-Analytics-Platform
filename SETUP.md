# 🛠️ Team Setup (Local)

0) ⚡ Prereqs
- 🐳 Docker Desktop (with WSL2 on Windows)
- 🟢 Node 18+ and npm
- 🛑 Ports free: 5432, 8080, 9000, 9001, 5173

1) 📥 Clone the repo & switch branch
```
git clone <your-repo-url>
cd Capstone-Business-Analytics
git checkout infra/postgresql-minio-keycloak-setup
```

2) 📝 Create env files from the examples
2a) 🗂️ Root env (for Docker services)
```
cp .env.example .env
```
The example is already filled for local use. If you change ports/usernames, keep them in sync later.

2b) 🖥️ Frontend env (for Vite/React)
```
cd frontend
cp .env.example .env
cd ..
```

3) 🚀 Start the infrastructure
From the repo root:
```
docker compose up -d
docker compose ps
```
If Postgres errors or you need a clean start:
```
docker compose down -v   # WARNING: wipes DB volume
docker compose up -d
```

4) 🔑 Initialize Keycloak (one-time)
Open http://localhost:8080
Login with:
- user: admin
- pass: password (from root .env)

Create Realm → name: booklatte
Create Client → name/ID: frontend
- Client type: OpenID Connect
- Client authentication: OFF
- Standard flow: ON (leave others OFF)
- Valid redirect URIs: http://localhost:5173/*
- Web origins: http://localhost:5173
(Optional) Root URL / Home URL: http://localhost:5173

Create a test user
- Users → Add user → username: testuser
- Credentials → set a password (uncheck Temporary), save

5) 📦 Prepare MinIO buckets
Open http://localhost:9001
Login with:
- user: booklatte
- pass: password (from root .env)

Create buckets:
- uploads
- models

6) 💻 Run the frontend (React + TypeScript)
```
cd frontend
npm install
npm run dev
```
Open http://localhost:5173
→ you should be redirected to Keycloak.
Login with testuser and the password you set. You’ll land back in the app.

If you’re not redirected:
Ensure frontend/.env contains:
```
VITE_KEYCLOAK_URL=http://localhost:8080
VITE_KEYCLOAK_REALM=booklatte
VITE_KEYCLOAK_CLIENT_ID=frontend
```
In Keycloak client frontend: Standard flow ON, valid redirect URI & web origins as above.
Restart the dev server after changing env: Ctrl+C → npm run dev.

7) 🗄️ (Optional) Verify databases
```
docker exec -it $(docker ps -qf "name=postgres") psql -U booklatte -d postgres -c "\l" | grep -E "booklatte|keycloak"
```
You should see booklatte and keycloak DBs (created by db/init/01-create-dbs.sql).

8) 🔜 What’s next (for the team)
- ETL: point Python scripts to read from MinIO uploads/ and load cleaned data into Postgres booklatte.
- Dash app: add a Dash service (8050) and embed in React (iframe) for interactive analytics.
- Upload API (later): small FastAPI service to generate presigned URLs so the browser can upload without exposing MinIO keys.

🧰 Troubleshooting quickies
- Port in use: stop whatever is using 5432/8080/9000/9001/5173 or change in .env and Keycloak client settings if you change 5173.
- Init SQL didn’t run: remove volume and restart:
```
docker compose down -v && docker compose up -d
```
- CORS / redirect errors: check Keycloak client Valid redirect URIs and Web origins.
- Vite not picking env: make sure the file is frontend/.env and variables start with VITE_.

✅ That’s it—following the steps above, any teammate can be up and running locally in a few minutes.