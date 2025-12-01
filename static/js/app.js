/**
 * TurisCaldas AI - JavaScript Principal
 * Chat con asistente de turismo para Caldas, Colombia
 */

// ==================== VARIABLES GLOBALES ====================
let currentConversationId = null;
let selectedFiles = [];
let isRecognizing = false;
let recognition = null;

// Constantes
const MAX_FILES = 3;
const MAX_FILE_SIZE = 16 * 1024 * 1024; // 16MB
const ALLOWED_EXTENSIONS = ['pdf', 'txt', 'docx'];

// ==================== INICIALIZACIÓN ====================
document.addEventListener('DOMContentLoaded', () => {
    initChat();
    initFileUpload();
    initSpeechRecognition();
    initModals();
    initNavigation();
});

// ==================== CHAT ====================
function initChat() {
    const chatForm = document.getElementById('chatForm');
    const chatInput = document.getElementById('chatInput');
    
    if (!chatForm) return;
    
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const text = chatInput.value.trim();
        if (!text) return;
        
        // Agregar mensaje del usuario
        appendMessage('user', text);
        chatInput.value = '';
        
        // Mostrar indicador de carga
        const loadingId = showLoading();
        
        try {
            const startTime = Date.now();
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                body: `message=${encodeURIComponent(text)}`
            });
            
            const data = await response.json();
            const responseTime = Date.now() - startTime;
            
            // Quitar loading
            removeLoading(loadingId);
            
            // Mostrar respuesta
            const cacheIcon = data.cached ? ' ⚡' : '';
            appendMessage('bot', data.response || 'Lo siento, no pude procesar tu consulta.', cacheIcon);
            
            // Mostrar fuentes si existen
            if (data.sources && data.sources.length > 0) {
                showSources(data.sources);
            }
            
            // Actualizar badge de confianza
            updateConfidenceBadge(data.confidence);
            
        } catch (error) {
            console.error('Error en chat:', error);
            removeLoading(loadingId);
            appendMessage('bot', 'Error de conexión. Por favor intenta de nuevo.');
        }
    });
}

function appendMessage(type, text, icon = '') {
    const chatMessages = document.getElementById('chatMessages');
    if (!chatMessages) return;
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${type}`;
    
    const avatar = type === 'bot' 
        ? '<div class="avatar bot-avatar">🦜</div>'
        : '<div class="avatar user-avatar">👤</div>';
    
    const sender = type === 'bot' ? 'TurisCaldas AI' : 'Tú';
    
    messageDiv.innerHTML = `
        ${avatar}
        <div class="message-content">
            <div class="message-header">${sender}${icon}</div>
            <div class="message-text">${escapeHtml(text)}</div>
        </div>
    `;
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function showLoading() {
    const chatMessages = document.getElementById('chatMessages');
    const loadingId = 'loading-' + Date.now();
    
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'chat-message bot loading';
    loadingDiv.id = loadingId;
    loadingDiv.innerHTML = `
        <div class="avatar bot-avatar">🦜</div>
        <div class="message-content">
            <div class="message-header">TurisCaldas AI</div>
            <div class="typing-indicator">
                <span></span><span></span><span></span>
            </div>
        </div>
    `;
    
    chatMessages.appendChild(loadingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    return loadingId;
}

function removeLoading(loadingId) {
    const loading = document.getElementById(loadingId);
    if (loading) loading.remove();
}

function showSources(sources) {
    const sourcesContainer = document.getElementById('sourcesContainer');
    if (!sourcesContainer) return;
    
    sourcesContainer.innerHTML = '<h4>📚 Fuentes consultadas:</h4>';
    sources.forEach((source, idx) => {
        const sourceDiv = document.createElement('div');
        sourceDiv.className = 'source-item';
        sourceDiv.innerHTML = `
            <strong>Fuente ${idx + 1}:</strong> ${escapeHtml(source.source || 'Documento')}
            ${source.page ? ` (pág. ${source.page})` : ''}
        `;
        sourcesContainer.appendChild(sourceDiv);
    });
    sourcesContainer.style.display = 'block';
}

function updateConfidenceBadge(confidence) {
    const badge = document.getElementById('confidenceBadge');
    if (!badge) return;
    
    badge.textContent = confidence ? `Confianza: ${confidence}` : '';
    badge.className = `confidence-badge ${confidence || ''}`;
}

// ==================== UPLOAD DE ARCHIVOS ====================
function initFileUpload() {
    const uploadForm = document.getElementById('uploadForm');
    const fileInput = document.getElementById('fileInput');
    
    if (!uploadForm || !fileInput) return;
    
    fileInput.addEventListener('change', (e) => {
        const newFiles = Array.from(e.target.files);
        
        if (selectedFiles.length + newFiles.length > MAX_FILES) {
            showUploadStatus(`⚠️ Máximo ${MAX_FILES} archivos`, 'warning');
            return;
        }
        
        newFiles.forEach(file => {
            const isDuplicate = selectedFiles.some(f => f.name === file.name);
            if (!isDuplicate) {
                selectedFiles.push(file);
            }
        });
        
        updateFilesList();
        e.target.value = '';
    });
    
    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const validFiles = selectedFiles.filter(f => isValidFile(f) && f.size <= MAX_FILE_SIZE);
        if (validFiles.length === 0) {
            showUploadStatus('❌ No hay archivos válidos', 'error');
            return;
        }
        
        const formData = new FormData();
        validFiles.forEach(file => formData.append('files', file));
        
        const uploadBtn = document.getElementById('uploadBtn');
        uploadBtn.disabled = true;
        uploadBtn.textContent = 'Procesando...';
        
        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (data.success) {
                showUploadStatus(`✅ ${data.message}`, 'success');
                selectedFiles = [];
                updateFilesList();
            } else {
                showUploadStatus(`❌ ${data.message}`, 'error');
            }
        } catch (error) {
            showUploadStatus('❌ Error de conexión', 'error');
        }
        
        uploadBtn.disabled = false;
        uploadBtn.textContent = 'Subir documentos';
    });
}

function updateFilesList() {
    const filesList = document.getElementById('filesList');
    const uploadBtn = document.getElementById('uploadBtn');
    if (!filesList) return;
    
    filesList.innerHTML = '';
    
    selectedFiles.forEach((file, index) => {
        const isValid = isValidFile(file);
        const isTooLarge = file.size > MAX_FILE_SIZE;
        
        const fileDiv = document.createElement('div');
        fileDiv.className = `file-item ${(!isValid || isTooLarge) ? 'invalid' : ''}`;
        fileDiv.innerHTML = `
            <span class="file-name">📄 ${escapeHtml(file.name)}</span>
            <span class="file-size">${formatFileSize(file.size)}</span>
            <button type="button" class="file-remove" onclick="removeFile(${index})">✕</button>
        `;
        filesList.appendChild(fileDiv);
    });
    
    const validCount = selectedFiles.filter(f => isValidFile(f) && f.size <= MAX_FILE_SIZE).length;
    if (uploadBtn) {
        uploadBtn.disabled = validCount === 0;
    }
}

function removeFile(index) {
    selectedFiles.splice(index, 1);
    updateFilesList();
}

function isValidFile(file) {
    const ext = file.name.split('.').pop().toLowerCase();
    return ALLOWED_EXTENSIONS.includes(ext);
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

function showUploadStatus(message, type) {
    const status = document.getElementById('uploadStatus');
    if (status) {
        status.textContent = message;
        status.className = `upload-status ${type}`;
    }
}

// ==================== RECONOCIMIENTO DE VOZ ====================
function initSpeechRecognition() {
    const micBtn = document.getElementById('micBtn');
    if (!micBtn) return;
    
    if ('SpeechRecognition' in window || 'webkitSpeechRecognition' in window) {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        recognition = new SpeechRecognition();
        recognition.lang = 'es-CO';
        recognition.interimResults = false;
        
        recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            const chatInput = document.getElementById('chatInput');
            if (chatInput) {
                chatInput.value = transcript;
                document.getElementById('chatForm').dispatchEvent(new Event('submit'));
            }
        };
        
        recognition.onend = () => {
            isRecognizing = false;
            micBtn.classList.remove('listening');
        };
        
        micBtn.addEventListener('click', toggleVoice);
    } else {
        micBtn.style.display = 'none';
    }
}

function toggleVoice() {
    const micBtn = document.getElementById('micBtn');
    
    if (isRecognizing) {
        recognition.stop();
        micBtn.classList.remove('listening');
    } else {
        recognition.start();
        micBtn.classList.add('listening');
    }
    isRecognizing = !isRecognizing;
}

// ==================== MODALES ====================
function initModals() {
    // Modal de Chat
    const chatModal = document.getElementById('chatModal');
    const openChatBtn = document.getElementById('openChatBtn');
    const closeChatBtn = document.getElementById('closeChatBtn');
    
    if (openChatBtn && chatModal) {
        openChatBtn.addEventListener('click', () => {
            chatModal.classList.add('active');
            document.body.style.overflow = 'hidden';
        });
    }
    
    if (closeChatBtn && chatModal) {
        closeChatBtn.addEventListener('click', () => {
            chatModal.classList.remove('active');
            document.body.style.overflow = '';
        });
    }
    
    // Modal de Historial
    const historyBtn = document.getElementById('historyBtn');
    const historyModal = document.getElementById('historyModal');
    const closeHistoryBtn = document.querySelector('.close-history');
    
    if (historyBtn && historyModal) {
        historyBtn.addEventListener('click', loadHistory);
    }
    
    if (closeHistoryBtn) {
        closeHistoryBtn.addEventListener('click', () => {
            historyModal.style.display = 'none';
        });
    }
    
    // Cerrar modales con click afuera
    window.addEventListener('click', (e) => {
        if (e.target === historyModal) {
            historyModal.style.display = 'none';
        }
        if (e.target === chatModal) {
            chatModal.classList.remove('active');
            document.body.style.overflow = '';
        }
    });
}

async function loadHistory() {
    const historyModal = document.getElementById('historyModal');
    const historyList = document.getElementById('historyList');
    
    if (!historyModal || !historyList) return;
    
    historyModal.style.display = 'flex';
    historyList.innerHTML = '<p>Cargando...</p>';
    
    try {
        const response = await fetch('/history');
        const data = await response.json();
        
        if (data.history && data.history.length > 0) {
            historyList.innerHTML = '';
            data.history.forEach(conv => {
                const div = document.createElement('div');
                div.className = 'history-item';
                const date = new Date(conv.timestamp).toLocaleString('es-CO');
                div.innerHTML = `
                    <div class="history-question">${escapeHtml(conv.question)}</div>
                    <div class="history-response">${escapeHtml(conv.response)}</div>
                    <div class="history-meta">${date}</div>
                `;
                historyList.appendChild(div);
            });
        } else {
            historyList.innerHTML = '<p>No hay conversaciones previas.</p>';
        }
    } catch (error) {
        historyList.innerHTML = '<p>Error al cargar historial.</p>';
    }
}

// ==================== NAVEGACIÓN ====================
function initNavigation() {
    // Header scroll effect
    const header = document.querySelector('.header');
    if (header) {
        window.addEventListener('scroll', () => {
            if (window.scrollY > 50) {
                header.classList.add('scrolled');
            } else {
                header.classList.remove('scrolled');
            }
        });
    }
    
    // Mobile menu toggle
    const menuToggle = document.getElementById('menuToggle');
    const navLinks = document.querySelector('.nav-links');
    
    if (menuToggle && navLinks) {
        menuToggle.addEventListener('click', () => {
            navLinks.classList.toggle('active');
            menuToggle.classList.toggle('active');
        });
    }
}

// ==================== UTILIDADES ====================
function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Función para búsqueda desde el Hero
function searchFromHero() {
    const searchInput = document.getElementById('heroSearch');
    if (searchInput && searchInput.value.trim()) {
        openChatWithMessage(searchInput.value.trim());
    }
}

function openChatWithMessage(message) {
    const chatModal = document.getElementById('chatModal');
    const chatInput = document.getElementById('chatInput');
    
    if (chatModal && chatInput) {
        chatModal.classList.add('active');
        document.body.style.overflow = 'hidden';
        chatInput.value = message;
        
        // Pequeño delay para asegurar que el modal esté visible
        setTimeout(() => {
            document.getElementById('chatForm').dispatchEvent(new Event('submit'));
        }, 100);
    }
}

// Scroll suave para navegación
function scrollToSection(sectionId) {
    const section = document.getElementById(sectionId);
    if (section) {
        section.scrollIntoView({ behavior: 'smooth' });
    }
}

// Nueva conversación
function nuevaConversacion() {
    const chatMessages = document.getElementById('chatMessages');
    if (chatMessages) {
        chatMessages.innerHTML = '';
        appendMessage('bot', '¡Hola de nuevo! 🦜 Empecemos una nueva aventura en Caldas.\n\nCuéntame: ¿De dónde vienes, cuántos son y qué les gustaría conocer?');
    }
    currentConversationId = null;
}

// Filtrar aves por categoría
function filterBirds(category) {
    const cards = document.querySelectorAll('#aviturismo .experience-card');
    const buttons = document.querySelectorAll('.avi-cat-btn');
    
    // Actualizar botones activos
    buttons.forEach(btn => btn.classList.remove('active'));
    event.target.classList.add('active');
    
    // Filtrar cards
    cards.forEach(card => {
        if (category === 'all' || card.dataset.category === category) {
            card.style.display = 'block';
            card.style.animation = 'fadeIn 0.3s ease';
        } else {
            card.style.display = 'none';
        }
    });
}

// ==================== REGISTRO DE NEGOCIOS ====================
function openRegistroModal() {
    const modal = document.getElementById('registroModal');
    if (modal) {
        modal.classList.add('active');
        document.body.style.overflow = 'hidden';
        
        // Resetear formulario
        document.getElementById('registroNegocioForm').style.display = 'block';
        document.getElementById('registroExito').style.display = 'none';
        document.getElementById('registroNegocioForm').reset();
    }
}

function closeRegistroModal() {
    const modal = document.getElementById('registroModal');
    if (modal) {
        modal.classList.remove('active');
        document.body.style.overflow = '';
    }
}

// Inicializar formulario de registro
document.addEventListener('DOMContentLoaded', () => {
    const registroForm = document.getElementById('registroNegocioForm');
    
    if (registroForm) {
        registroForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            // Recopilar datos del formulario
            const formData = {
                nombre: document.getElementById('negocioNombre').value,
                tipo: document.getElementById('negocioTipo').value,
                municipio: document.getElementById('negocioMunicipio').value,
                precio: document.getElementById('negocioPrecio').value,
                contacto: document.getElementById('negocioContacto').value,
                telefono: document.getElementById('negocioTelefono').value,
                email: document.getElementById('negocioEmail').value,
                descripcion: document.getElementById('negocioDescripcion').value,
                fecha: new Date().toISOString()
            };
            
            try {
                // Enviar al servidor
                const response = await fetch('/registrar-negocio', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(formData)
                });
                
                const data = await response.json();
                
                if (data.success) {
                    // Mostrar mensaje de éxito
                    document.getElementById('registroNegocioForm').style.display = 'none';
                    document.getElementById('registroExito').style.display = 'block';
                } else {
                    alert('Error al enviar: ' + (data.message || 'Intenta de nuevo'));
                }
            } catch (error) {
                // Si no hay endpoint, igual mostrar éxito (para demo)
                console.log('Datos del negocio:', formData);
                document.getElementById('registroNegocioForm').style.display = 'none';
                document.getElementById('registroExito').style.display = 'block';
            }
        });
    }
    
    // Cerrar modal al hacer clic afuera
    const registroModal = document.getElementById('registroModal');
    if (registroModal) {
        registroModal.addEventListener('click', (e) => {
            if (e.target === registroModal) {
                closeRegistroModal();
            }
        });
    }
});

// Abrir chat con mensaje de bienvenida
function openChat() {
    const chatModal = document.getElementById('chatModal');
    const chatMessages = document.getElementById('chatMessages');
    
    if (chatModal) {
        chatModal.classList.add('active');
        document.body.style.overflow = 'hidden';
        
        // Mensaje de bienvenida si está vacío
        if (chatMessages && chatMessages.children.length === 0) {
            appendMessage('bot', '¡Hola! 🦜 Soy TurisCaldas AI, tu guía turístico virtual para Caldas y el Eje Cafetero.\n\nPara darte las mejores recomendaciones, cuéntame:\n• ¿De dónde nos visitas?\n• ¿Cuántas personas viajan?\n• ¿Qué tipo de turismo te interesa? (café ☕, termales ♨️, aves 🐦, aventura 🏔️, naturaleza 🌿)');
        }
    }
}
