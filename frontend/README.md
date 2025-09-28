# Capstone Frontend

This is the **React frontend** for the Capstone Project:  
**Data-Driven Product Bundling Strategy and Pricing Optimization Using Market Basket Analysis and Prescriptive Analytics**

The frontend provides:
- 🔑 **Login Page** using Keycloak (OpenID Connect authentication).  
- 📂 **File Upload** for sales and transaction datasets to MinIO.  
- 📊 **Embedded Dash App** for interactive analytics (e.g., Market Basket Analysis, Price Elasticity of Demand, Exponential Smoothing).  
- 📡 **API Integration** with the backend ETL/ML services running in Docker Compose.  

---

## 📦 Folder Structure

```
frontend/
├─ public/            # static assets
├─ src/
│  ├─ components/     # reusable UI components
│  ├─ pages/          # Login, Dashboard, Upload
│  ├─ services/       # Keycloak and API helpers
│  ├─ App.tsx         # main app component
│  └─ main.tsx        # entrypoint
├─ .env               # environment variables
├─ package.json
└─ README.md
```

---

## ⚙️ Environment Variables

Create a `.env` file in the `frontend/` directory:

```ini
VITE_KEYCLOAK_URL=http://localhost:8080
VITE_KEYCLOAK_REALM=capstone
VITE_KEYCLOAK_CLIENT_ID=frontend

VITE_DASH_URL=http://localhost:8050
VITE_MINIO_CONSOLE=http://localhost:9001
```

---

## Development

Install dependencies:

```
npm install
```

Start the Vite dev server:

```
npm run dev
```

Visit the app:

http://localhost:5173
