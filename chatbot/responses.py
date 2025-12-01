# -*- coding: utf-8 -*-
"""
Sistema de respuestas predefinidas para TurisCaldas AI
Asistente conversacional orientado a turistas que visitan el departamento de Caldas
"""

# Respuestas base del sistema (turismo)
RESPUESTAS_BASE = {
    "saludo": [
        "¡Hola! Soy TurisCaldas, tu asistente de viajes en Caldas. ¿En qué puedo ayudarte hoy?",
        "¡Bienvenido/a a TurisCaldas! ¿Buscas actividades, alojamiento o recomendaciones locales?",
        "¡Hola! Soy tu guía virtual para explorar Caldas. Dime qué te interesa: café, termales, naturaleza u otro plan?",
        "¡Saludos! Estoy aquí para ayudarte a planear tu visita en Caldas. ¿Prefieres recomendaciones por presupuesto, tipo de actividad o ubicación?"
    ],
    "saludo_completo": [
        "¡Hola! Soy TurisCaldas, un asistente que te ayuda a descubrir rutas, alojamientos y experiencias en Caldas. ¿Cómo te gustaría empezar?",
        "Encantado/a de ayudarte a planear tu viaje en Caldas: puedo sugerir itinerarios, restaurantes, hospedajes y actividades según tus intereses.",
        "Bienvenido/a: dime cuántos días tienes y tus intereses y te propongo un plan personalizado por zonas y presupuesto."
    ],
    "despedida": [
        "¡Que disfrutes tu viaje por Caldas! Si necesitas más recomendaciones, aquí estaré. ¡Buen viaje!",
        "Gracias por usar TurisCaldas. Vuelve cuando quieras para ajustar tu itinerario o descubrir nuevas experiencias.",
        "¡Listo! Espero que la información te sea útil. Avísame si quieres reservar o ampliar el plan."
    ],
    "no_entiendo": [
        "No entendí completamente tu consulta. ¿Puedes dar más detalles sobre lo que buscas (ej. tipo de actividad, presupuesto, fechas)?",
        "Necesito un poco más de información para ayudarte mejor: ¿viajas solo, en pareja o en familia? ¿Cuántos días estarás?",
        "Por favor, especifica si buscas alojamiento, actividades, rutas o recomendaciones gastronómicas para que pueda ayudarte mejor."
    ]
}

# Respuestas cuando no se encuentra información (turismo)
RESPUESTAS_NO_ENCONTRADO = {
    "hoteles": [
        "🏨 No encontré información clara sobre alojamientos que coincidan con tus criterios. Te sugiero:",
        "- Ampliar el rango de presupuesto o la localidad",
        "- Verificar disponibilidad en las fechas indicadas",
        "- ¿Quieres que busque opciones cercanas a una ciudad en particular (Manizales, Salamina, Aguadas)?"
    ],
    "atractivos": [
        "📍 No hallé detalles sobre ese atractivo turístico en la información disponible. Puedes:",
        "- Proporcionar el nombre exacto del sitio o municipio",
        "- Consultar si está en eventos o temporadas específicas",
        "- ¿Deseas alternativas similares cerca de tu ubicación?"
    ],
    "general": [
        "🔎 No encontré datos relevantes para tu consulta. Para ayudarte mejor, puedes:",
        "- Especificar fecha, lugar o tipo de experiencia (aventura, gastronómico, cultural)",
        "- Subir información adicional o consultar por municipios específicos",
        "¿Quieres que te proponga planes generales para 1, 2 o 3 días?"
    ],
    "itinerario": [
        "🗺️ No hay información suficiente para generar un itinerario completo. Recomendaciones:",
        "- Indica duración del viaje y punto de inicio",
        "- Especifica intereses y presupuesto",
        "¿Quieres que proponga un itinerario básico según tus preferencias?"
    ]
}

# Respuestas para diferentes tipos de consultas (contexto turístico)
RESPUESTAS_CONTEXTUALES = {
    "carga_documentos": [
        "Perfecto, he recibido la información (guías, folletos o datos). Procesaré el contenido y podré responder consultas sobre destinos y servicios.",
        "Información cargada correctamente. Ahora puedo sugerir itinerarios, alojamientos y actividades basadas en los datos.",
        "Gracias, ya puedo usar estos datos para ofrecer recomendaciones locales y generar un itinerario básico."
    ],
    "analisis_riesgo": [
        "⚠️ **Aviso de condiciones**: Detecté factores que pueden afectar tu viaje (clima, cierres temporales, temporada alta):",
        "🔍 **Consideraciones de viaje**: Revisa disponibilidad y condiciones de accesibilidad en las rutas propuestas:",
        "📊 **Recomendación práctica**: Te propongo alternativas en caso de condiciones adversas:"
    ],
    "informacion_atractivo": [
        "📍 **Información del atractivo**: Según los datos disponibles:",
        "🕒 **Horarios y recomendaciones**: Ten en cuenta horarios, temporada y recomendaciones locales:",
        "📸 **Consejos prácticos**: Mejor época para la visita, qué llevar y cómo llegar:",
    ],
    "recomendaciones": [
        "💡 **Sugerencias de viaje**: Basado en tus preferencias, te recomiendo:",
        "🎯 **Itinerario sugerido**: Un plan día a día según intereses y tiempo disponible:",
        "📋 **Checklist de viaje**: Elementos y recomendaciones para tu experiencia en Caldas:"
    ]
}

# Respuestas de error (turismo)
RESPUESTAS_ERROR = {
    "error_procesamiento": [
        "⚠️ Disculpa, ocurrió un error técnico al procesar tu solicitud. Intenta nuevamente en unos segundos.",
        "🔧 Hubo un problema al generar la respuesta. Por favor vuelve a intentarlo o especifica tu consulta de otra forma.",
        "❌ Error temporal en el servicio. Si el problema persiste, crea un issue en el repositorio."
    ],
    "documento_no_valido": [
        "📄 Formato no compatible. Aceptamos PDF, DOC, DOCX y TXT para procesar información turística.",
        "🚫 El archivo no pudo ser procesado. Verifica que el documento esté legible y vuelva a intentarlo.",
        "⚠️ Archivo no válido: sube un PDF o documento de texto estándar para extraer información."
    ],
    "limite_tamaño": [
        "📏 El archivo excede el límite de tamaño. Por favor divide la información en archivos más pequeños.",
        "⚡ Para un procesamiento eficiente, sube archivos menores a 16MB o segmenta el contenido por secciones.",
        "📊 El tamaño del archivo supera los límites técnicos. Puedes subir por municipios o por tipo de información."
    ]
}

# Respuestas para mantener el rol (turismo)
RESPUESTAS_ROL = {
    "fuera_contexto": [
        "🧭 Soy TurisCaldas, un asistente especializado en turismo local. Puedo ayudar con rutas, alojamientos, actividades y recomendaciones culturales.",
        "🌄 Mi enfoque es turístico: planificación de itinerarios, sugerencias gastronómicas y logística de viaje en Caldas. ¿En qué te puedo apoyar?",
        "📣 Estoy aquí para facilitar tu experiencia turística en Caldas, conectar con prestadores locales y sugerir planes según tus intereses."
    ],
    "aclaracion_rol": [
        "Soy tu asistente de viajes para Caldas. Puedo:",
        "• Sugerir itinerarios y rutas",
        "• Recomendar alojamientos y restaurantes",
        "• Proponer actividades según intereses (café, termales, aventura)",
        "• Dar información práctica: horarios, precios y cómo llegar",
        "¿Qué necesitas planear hoy?"
    ]
}

# Respuestas con diferentes niveles de confianza (turismo)
RESPUESTAS_CONFIANZA = {
    "alta": [
        "✅ **Información verificada**: Según los datos disponibles, esta recomendación cumple tus criterios.",
        "🎯 **Alta confianza**: Esta opción es apropiada según preferencias y disponibilidad conocida.",
        "📌 **Recomendación segura**: Basado en fuentes y datos, esta es una buena elección."
    ],
    "media": [
        "📋 **Confianza media**: Hay información parcial o variables (clima, temporada) que podrían afectar la elección.",
        "🔍 **Evaluación preliminar**: Requiere confirmación de disponibilidad o condiciones locales.",
        "🟡 **Sugerencia tentativa**: Útil como referencia, pero verifica horarios y reservas."
    ],
    "baja": [
        "⚠️ **Baja confianza**: Información incompleta o no verificada. Recomendable confirmar antes de viajar.",
        "🔎 **Consulta adicional**: Necesito más datos (fechas, localidad exacta) para ofrecer una recomendación confiable.",
        "ℹ️ **Referencia**: Úsalo como punto de partida y verifica con prestadores locales."
    ]
}

def get_respuesta_by_tipo(tipo, subtipo="general"):
    """
    Obtiene una respuesta aleatoria del tipo especificado
    """
    import random
    
    if tipo in RESPUESTAS_BASE:
        return random.choice(RESPUESTAS_BASE[tipo])
    elif tipo == "no_encontrado":
        if subtipo in RESPUESTAS_NO_ENCONTRADO:
            return "\n".join(RESPUESTAS_NO_ENCONTRADO[subtipo])
        return "\n".join(RESPUESTAS_NO_ENCONTRADO["general"])
    elif tipo in RESPUESTAS_CONTEXTUALES:
        return random.choice(RESPUESTAS_CONTEXTUALES[tipo])
    elif tipo in RESPUESTAS_ERROR:
        return random.choice(RESPUESTAS_ERROR[tipo])
    elif tipo in RESPUESTAS_ROL:
        return random.choice(RESPUESTAS_ROL[tipo])
    elif tipo == "confianza":
        if subtipo in RESPUESTAS_CONFIANZA:
            return random.choice(RESPUESTAS_CONFIANZA[subtipo])
    
    return "Soy TurisCaldas, tu asistente de viajes en Caldas. Dime qué necesitas: recomendaciones, itinerarios o información local."

def get_respuesta_no_encontrado_inteligente(pregunta):
    """
    Determina el mejor tipo de respuesta NO_ENCONTRADO basado en la pregunta
    """
    pregunta_lower = pregunta.lower()
    
    palabras_hoteles = ["hotel", "alojamiento", "hostal", "hospedaje", "cabaña"]
    palabras_atractivos = ["atractivo", "termales", "ruta", "sendero", "parque", "sitio"]
    palabras_itinerario = ["itinerario", "plan", "día", "días", "ruta recomendada"]

    if any(palabra in pregunta_lower for palabra in palabras_hoteles):
        return get_respuesta_by_tipo("no_encontrado", "hoteles")
    elif any(palabra in pregunta_lower for palabra in palabras_atractivos):
        return get_respuesta_by_tipo("no_encontrado", "atractivos")
    elif any(palabra in pregunta_lower for palabra in palabras_itinerario):
        return get_respuesta_by_tipo("no_encontrado", "itinerario")
    else:
        return get_respuesta_by_tipo("no_encontrado", "general")