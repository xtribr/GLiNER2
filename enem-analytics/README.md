# X-TRI Escolas - Ranking ENEM Analytics

Plataforma de análise de desempenho escolar no ENEM com inteligência artificial.

## Stack

- **Backend**: FastAPI (Python 3.11+)
- **Frontend**: Next.js 16 + React 19 + Tailwind CSS
- **ML**: XGBoost, scikit-learn, GLiNER
- **Database**: SQLite (dados ENEM)

## Estrutura

```
enem-analytics/
├── backend/
│   ├── api/
│   │   ├── main.py              # FastAPI app
│   │   └── routes/              # API endpoints
│   ├── data/                    # Dados processados
│   ├── ml/                      # Modelos ML
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── app/                 # Next.js pages
│   │   ├── components/          # React components
│   │   └── lib/                 # Utils e API client
│   └── package.json
└── README.md
```

## Deploy no Railway

### Backend (FastAPI)

1. Criar novo serviço no Railway
2. Conectar ao repositório Git
3. Configurar:
   - **Root Directory**: `enem-analytics/backend`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`

### Frontend (Next.js)

1. Criar novo serviço no Railway
2. Conectar ao repositório Git
3. Configurar:
   - **Root Directory**: `enem-analytics/frontend`
   - **Build Command**: `pnpm install && pnpm build`
   - **Start Command**: `pnpm start`

4. Variáveis de ambiente:
   ```
   NEXT_PUBLIC_API_URL=https://seu-backend.railway.app
   ```

## Variáveis de Ambiente

### Backend
```env
PORT=8000
PIONEER_API_KEY=sua_chave_pioneer  # Para GLiNER
```

### Frontend
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Desenvolvimento Local

### Backend
```bash
cd enem-analytics/backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000
```

### Frontend
```bash
cd enem-analytics/frontend
pnpm install
pnpm dev
```

## Funcionalidades

- **Ranking de Escolas**: Visualização do ranking ENEM por escola
- **Análise de Habilidades**: Comparação com média nacional
- **Predições ML**: Projeção de notas TRI futuras
- **Diagnóstico**: Identificação de pontos fracos
- **Clustering**: Agrupamento por perfil de habilidades
- **Recomendações**: Plano de melhoria personalizado
- **BrainX Insights**: Análise inteligente com IA

## API Endpoints

### Schools
- `GET /api/schools` - Lista escolas com filtros
- `GET /api/schools/{codigo_inep}` - Detalhes da escola
- `GET /api/schools/{codigo_inep}/history` - Histórico de notas
- `GET /api/schools/{codigo_inep}/skills` - Habilidades da escola

### ML Analytics
- `GET /api/predictions/{codigo_inep}` - Predições de notas
- `GET /api/diagnosis/{codigo_inep}` - Diagnóstico completo
- `GET /api/clusters/{codigo_inep}/cluster` - Cluster da escola
- `GET /api/recommendations/{codigo_inep}` - Recomendações

### BrainX
- `GET /api/gliner/{codigo_inep}/insights` - Insights de IA
- `GET /api/tri-lists/{codigo_inep}/{area}` - Listas TRI por área

## Licença

Proprietário - X-TRI
