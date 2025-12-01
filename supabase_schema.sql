-- ==========================================================
-- Esquema de base de datos para TurisCaldas AI
-- Asistente turístico inteligente para Caldas, Colombia
-- Ejecutar en Supabase SQL Editor
-- ==========================================================

-- Tabla para almacenar información de documentos turísticos
CREATE TABLE IF NOT EXISTS documents (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  filename VARCHAR(255) NOT NULL,
  file_path VARCHAR(500),
  corpus_id VARCHAR(64) NOT NULL,
  doc_type VARCHAR(50) DEFAULT 'general', -- 'hotel', 'restaurante', 'atractivo', 'ruta', 'evento'
  status VARCHAR(50) DEFAULT 'processing',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Tabla para almacenar conversaciones con turistas
CREATE TABLE IF NOT EXISTS conversations (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_question TEXT NOT NULL,
  bot_response TEXT NOT NULL,
  sources JSONB DEFAULT '[]',
  confidence VARCHAR(20),
  practical_info JSONB DEFAULT '[]', -- horarios, precios, cómo llegar
  recommendations JSONB DEFAULT '[]',
  corpus_id VARCHAR(64),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  session_id UUID
);

-- Tabla para métricas y analytics de turismo
CREATE TABLE IF NOT EXISTS analytics (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  event_type VARCHAR(50) NOT NULL, -- 'consulta', 'busqueda_hotel', 'busqueda_ruta', 'upload'
  query_category VARCHAR(50), -- 'hospedaje', 'gastronomia', 'aventura', 'cultura', 'termales'
  municipality VARCHAR(100), -- municipio consultado
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Índices para mejor rendimiento
CREATE INDEX IF NOT EXISTS idx_documents_corpus_id ON documents(corpus_id);
CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at);
CREATE INDEX IF NOT EXISTS idx_conversations_corpus_id ON conversations(corpus_id);
CREATE INDEX IF NOT EXISTS idx_conversations_created_at ON conversations(created_at);
CREATE INDEX IF NOT EXISTS idx_conversations_session_id ON conversations(session_id);
CREATE INDEX IF NOT EXISTS idx_analytics_event_type ON analytics(event_type);
CREATE INDEX IF NOT EXISTS idx_analytics_created_at ON analytics(created_at);

-- RLS (Row Level Security) políticas
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE conversations ENABLE ROW LEVEL SECURITY;
ALTER TABLE analytics ENABLE ROW LEVEL SECURITY;

-- Política para permitir operaciones anónimas (ajustar según necesidades de seguridad)
CREATE POLICY "Allow anonymous access" ON documents FOR ALL USING (true);
CREATE POLICY "Allow anonymous access" ON conversations FOR ALL USING (true);
CREATE POLICY "Allow anonymous access" ON analytics FOR ALL USING (true);

-- Triggers para actualizar timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_documents_updated_at 
    BEFORE UPDATE ON documents 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();