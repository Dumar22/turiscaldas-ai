# 🌄 TurisCaldas AI - Asistente Inteligente para el Turismo en Caldas

**TurisCaldas AI** es un asistente virtual inteligente que utiliza IA avanzada para ofrecer recomendaciones personalizadas sobre turismo en Caldas. Ayuda a viajeros y turistas a descubrir destinos, hospedajes, restaurantes y actividades, conectándolos con prestadores de servicios locales mediante procesamiento de lenguaje natural y búsqueda semántica.

## 🚀 Despliegue en Producción

### Render.com (Recomendado)

1. Fork el repositorio en GitHub
2. Crear cuenta en [render.com](https://render.com)  
3. Nuevo Web Service → Conectar repositorio
4. Configurar variables de entorno (ver sección Configuración)
5. Desplegar automáticamente

> Archivos incluidos: `Procfile`, `build.sh`, `runtime.txt`. Ver `DEPLOYMENT.md` para detalles.

## ✨ Características Principales

- **RAG con FAISS**: Búsqueda semántica en documentos turísticos
- **Recomendaciones personalizadas**: Según preferencias del viajero (aventura, cultura, gastronomía, bienestar)
- **Base de datos Supabase**: Persistencia de conversaciones y documentos
- **Caché inteligente**: Respuestas rápidas con caché en memoria (5 min) y disco (1 hora)
- **Interfaz responsiva**: Diseño adaptativo para móviles y escritorio
- **Múltiples formatos**: Soporta PDF, TXT, DOCX

## 🧠 ¿Cómo funciona el Bot?

### Sistema RAG (Retrieval-Augmented Generation)

```bash
Documento → Chunks → Embeddings → FAISS → Búsqueda → GPT → Respuesta
```

1. **Cargar información**: El admin sube PDFs/TXT con datos turísticos (hoteles, rutas, atractivos)
2. **Vectorización**: El sistema divide en fragmentos y genera embeddings con OpenAI
3. **Almacenamiento**: Los vectores se guardan en FAISS (`vector_db/`)
4. **Consulta**: Cuando un turista pregunta, se busca contexto relevante
5. **Respuesta**: GPT genera respuesta basada en la información encontrada

### Sistema de Caché (respuestas rápidas)

| Nivel | TTL | Capacidad | Velocidad |
|-------|-----|-----------|-----------|
| Memoria | 5 min | 50 consultas | Ultra-rápido |
| Disco | 1 hora | 200 consultas | Rápido |

### Alimentar el Bot

Para que el bot tenga información de Caldas, sube documentos con:

- Guías turísticas de municipios
- Información de hoteles y restaurantes
- Datos del RNT (Registro Nacional de Turismo)
- Eventos y festividades
- Rutas y atractivos turísticos

## 🛠️ Stack Tecnológico

- **Backend**: Flask, Python 3.11+
- **IA**: OpenAI GPT-4o, LangChain, FAISS
- **Base de datos**: Supabase (PostgreSQL)
- **Frontend**: HTML5, CSS3, JavaScript

## 🚀 Instalación Local

### 1. Clonar y configurar entorno

```bash
git clone https://github.com/Dumar22/proyecto-final.git
cd proyecto-final
python -m venv env
source env/bin/activate  # Linux/Mac
# env\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. Configurar variables de entorno

```bash
cp .env.example .env
# Editar .env con tus credenciales:
# OPENAI_API_KEY=sk-tu-api-key
# SUPABASE_URL=https://tu-proyecto.supabase.co
# SUPABASE_ANON_KEY=tu-key
```

### 3. Configurar Supabase

1. Crear proyecto en [Supabase](https://supabase.com)
2. Ejecutar `supabase_schema.sql` en el Editor SQL
3. Obtener URL y Anon Key desde Project Settings > API

### 4. Ejecutar

```bash
python main.py
# Acceder a http://localhost:5000
```

## 📚 Uso

1. **Subir documentos**: Cargar información turística (guías, folletos, información de destinos)
2. **Consultar**: Hacer preguntas como:
   - "¿Qué actividades de aventura hay en Caldas?"
   - "Recomiéndame hoteles con termales"
   - "¿Cuál es la mejor ruta cafetera?"
3. **Ver historial**: Revisar conversaciones anteriores

## 📁 Estructura del Proyecto

```bash
proyecto-final/
├── main.py                # Aplicación Flask
├── requirements.txt       # Dependencias
├── .env.example          # Plantilla configuración
├── supabase_schema.sql   # Schema BD
├── chatbot/              # Módulos del bot
├── static/               # CSS y JS
├── templates/            # HTML
├── uploads/              # Documentos (generado)
└── vector_db/            # FAISS (generado)
```

## 🔧 API Endpoints

- `POST /upload` - Subir y procesar documento
- `POST /chat` - Procesar consulta del usuario
- `GET /history` - Obtener historial de conversaciones

## 📄 Licencia

MIT License. Ver `LICENSE` para detalles.

## 📞 Contacto

Para soporte o consultas, crear un issue en GitHub.

---

**Proyecto Talento Tech 2025 - MinTIC | Cohorte G339**

