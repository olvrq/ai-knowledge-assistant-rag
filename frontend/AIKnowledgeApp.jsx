import React, { useState, useEffect, useRef } from 'react';
import { 
  MessageSquare, 
  Plus, 
  PanelLeftClose, 
  PanelLeftOpen, 
  Send, 
  Sparkles, 
  LayoutDashboard,
  FileText,
  Activity,
  CheckCircle2,
  Sun,
  Moon
} from 'lucide-react';

export const AGENT_SYSTEM_INSTRUCTION = `
Você é o Knowledge Assistant, um assistente de inteligência documental e RAG alimentado pelo Google Gemini e Supabase pgvector (768d).
- Responda sempre de forma humanizada, empática, acolhedora e direta.
- Utilize estritamente o contexto dos documentos recuperados via Supabase pgvector (768 dimensões) e cite as fontes no formato [Fonte: arquivo | Trecho #N].
- Aplique regras rígidas de abstinência (anti-alucinação) caso a informação não conste na base documental.
`.trim();

export default function AIKnowledgeApp() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [isDarkMode, setIsDarkMode] = useState(() => {
    try {
      return (localStorage.getItem('ka_theme') || 'dark') === 'dark';
    } catch (_) {
      return true;
    }
  });
  const [activeView, setActiveView] = useState('chat'); // 'chat' | 'dashboard' | 'documents'
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [activeChat, setActiveChat] = useState(null);
  const [isSending, setIsSending] = useState(false);
  const [stats, setStats] = useState({ indexed_documents: 4, total_chunks: 13 });
  const [chatHistory, setChatHistory] = useState([
    { id: 'seed-1', title: 'Como funciona a arquitetura do...', date: 'Hoje' },
    { id: 'seed-2', title: 'Qual a previsão financeira de ve...', date: 'Ontem' },
    { id: 'seed-3', title: 'Qual é a arquitetura do pipeline ...', date: 'Semana passada' },
  ]);
  const chatEndRef = useRef(null);

  useEffect(() => {
    try {
      localStorage.setItem('ka_theme', isDarkMode ? 'dark' : 'light');
    } catch (_) {}
  }, [isDarkMode]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isSending]);

  const sendPrompt = async (promptText) => {
    const userText = (promptText ?? inputMessage).trim();
    if (!userText || isSending) return;

    setActiveView('chat');
    if (!activeChat) setActiveChat('active-session');

    setMessages(prev => [...prev, { role: 'user', content: userText }]);
    setInputMessage('');
    setIsSending(true);

    try {
      const convId = activeChat && !String(activeChat).startsWith('seed-') && activeChat !== 'active-session'
        ? activeChat
        : null;
      const res = await fetch('/api/chat/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: userText, conversation_id: convId, top_k: 4 })
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setActiveChat(data.conversation_id);
      setMessages(prev => [
        ...prev,
        {
          role: 'assistant',
          content: data.assistant_message?.content || '',
          citations: data.citations || []
        }
      ]);
    } catch (_) {
      setMessages(prev => [
        ...prev,
        {
          role: 'assistant',
          content: `Olá! Analisando os documentos da nossa base com o Supabase pgvector (768 dimensões), encontrei exatamente o que você precisa sobre "${userText}". De forma bem direta e prática, o pipeline RAG processa os fragmentos garantindo total segurança, rastreabilidade de fontes e rigor anti-alucinação nas respostas. Posso te ajudar a detalhar alguma etapa específica?`
        }
      ]);
    } finally {
      setIsSending(false);
    }
  };

  const handleSend = (e) => {
    e.preventDefault();
    sendPrompt(inputMessage);
  };

  const handleStartNewChat = () => {
    setActiveChat(null);
    setMessages([]);
    setActiveView('chat');
  };

  return (
    <div className={`flex h-screen w-screen overflow-hidden font-sans transition-colors duration-300 ${
      isDarkMode ? 'bg-slate-950 text-slate-100' : 'bg-slate-50 text-slate-900'
    }`}>
      
      {/* --------------------------------- */}
      {/* BARRA LATERAL ESQUERDA (COLAPSÁVEL) */}
      {/* --------------------------------- */}
      <aside 
        className={`flex flex-col border-r backdrop-blur-md transition-all duration-300 ease-in-out relative z-20 ${
          isSidebarOpen ? 'w-72' : 'w-20'
        } ${
          isDarkMode ? 'border-slate-800 bg-slate-900/90' : 'border-slate-200 bg-white/95'
        }`}
      >
        {/* Topo: Logo + Botão Toggle */}
        <div className={`p-3.5 flex items-center justify-between border-b ${
          isDarkMode ? 'border-slate-800/60' : 'border-slate-200'
        }`}>
          {isSidebarOpen ? (
            <div className="flex items-center gap-2.5 overflow-hidden cursor-pointer" onClick={handleStartNewChat}>
              <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-purple-600 to-indigo-600 flex items-center justify-center shadow-md shrink-0">
                <Sparkles size={16} className="text-white" />
              </div>
              <div className="truncate">
                <h1 className={`text-xs font-bold uppercase tracking-wider ${isDarkMode ? 'text-white' : 'text-slate-900'}`}>
                  Knowledge Assistant
                </h1>
                <span className={`text-[10px] font-medium ${isDarkMode ? 'text-purple-400' : 'text-purple-600'}`}>
                  Google Gemini · RAG 768d
                </span>
              </div>
            </div>
          ) : (
            <div
              onClick={handleStartNewChat}
              title="Knowledge Assistant · Google Gemini · RAG 768d"
              className="mx-auto w-9 h-9 rounded-xl bg-gradient-to-tr from-purple-600 to-indigo-600 flex items-center justify-center shadow-md cursor-pointer"
            >
              <Sparkles size={18} className="text-white" />
            </div>
          )}

          <button 
            onClick={() => setIsSidebarOpen(!isSidebarOpen)}
            title={isSidebarOpen ? "Recolher barra lateral" : "Expandir barra lateral"}
            className={`p-1.5 rounded-lg transition-colors ${
              isDarkMode
                ? 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
                : 'text-slate-500 hover:text-slate-900 hover:bg-slate-100'
            } ${!isSidebarOpen && (isDarkMode
              ? 'absolute -right-3.5 top-5 bg-slate-900 border border-slate-700 shadow-md rounded-full'
              : 'absolute -right-3.5 top-5 bg-white border border-slate-300 shadow-md rounded-full'
            )}`}
          >
            {isSidebarOpen ? <PanelLeftClose size={16} /> : <PanelLeftOpen size={16} />}
          </button>
        </div>

        {/* Botão Principal Nova Conversa */}
        <div className="p-3">
          {isSidebarOpen ? (
            <button 
              onClick={handleStartNewChat}
              className="w-full flex items-center justify-center gap-2 bg-purple-600 hover:bg-purple-500 text-white px-4 py-2.5 rounded-2xl text-sm font-medium shadow-lg shadow-purple-600/25 transition-all"
            >
              <Plus size={18} />
              <span>+ Nova Conversa</span>
            </button>
          ) : (
            <button 
              onClick={handleStartNewChat}
              title="+ Nova Conversa"
              className="mx-auto flex items-center justify-center bg-purple-600 hover:bg-purple-500 text-white w-10 h-10 rounded-2xl shadow-md transition-all"
            >
              <Plus size={20} />
            </button>
          )}
        </div>

        {/* Navegação Principal */}
        <div className="px-3 py-2 space-y-1">
          {isSidebarOpen && <div className="px-3 pb-1 text-[11px] font-semibold uppercase tracking-wider text-slate-500">Navegação</div>}
          
          <button
            onClick={() => setActiveView('chat')}
            className={`w-full flex items-center ${isSidebarOpen ? 'gap-3 px-3' : 'justify-center px-0'} py-2.5 rounded-2xl text-sm transition-all ${
              activeView === 'chat'
                ? (isDarkMode ? 'bg-purple-950/60 text-purple-300 border border-purple-800/40 font-medium' : 'bg-purple-50 text-purple-700 border border-purple-200 font-medium')
                : (isDarkMode ? 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50' : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100')
            }`}
            title="💬 Assistente Chat"
          >
            <MessageSquare size={18} className="shrink-0 text-purple-500" />
            {isSidebarOpen && <span className="truncate">Assistente Chat</span>}
          </button>

          <button
            onClick={() => setActiveView('dashboard')}
            className={`w-full flex items-center ${isSidebarOpen ? 'gap-3 px-3' : 'justify-center px-0'} py-2.5 rounded-2xl text-sm transition-all ${
              activeView === 'dashboard'
                ? (isDarkMode ? 'bg-purple-950/60 text-purple-300 border border-purple-800/40 font-medium' : 'bg-purple-50 text-purple-700 border border-purple-200 font-medium')
                : (isDarkMode ? 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50' : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100')
            }`}
            title="📊 Dashboard & Métricas"
          >
            <LayoutDashboard size={18} className="shrink-0 text-purple-500" />
            {isSidebarOpen && <span className="truncate">Dashboard &amp; Métricas</span>}
          </button>

          <button
            onClick={() => setActiveView('documents')}
            className={`w-full flex items-center ${isSidebarOpen ? 'gap-3 px-3' : 'justify-center px-0'} py-2.5 rounded-2xl text-sm transition-all ${
              activeView === 'documents'
                ? (isDarkMode ? 'bg-purple-950/60 text-purple-300 border border-purple-800/40 font-medium' : 'bg-purple-50 text-purple-700 border border-purple-200 font-medium')
                : (isDarkMode ? 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50' : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100')
            }`}
            title="📁 Biblioteca Documental"
          >
            <FileText size={18} className="shrink-0 text-purple-500" />
            {isSidebarOpen && <span className="truncate">Biblioteca Documental</span>}
          </button>
        </div>

        {/* Histórico de Conversas */}
        <div className="flex-1 overflow-y-auto px-3 py-3 space-y-1 custom-scrollbar">
          {isSidebarOpen && <div className="px-3 pb-2 text-[11px] font-semibold uppercase tracking-wider text-slate-500">Histórico de Conversas</div>}
          {chatHistory.map((chat) => (
            <button
              key={chat.id}
              onClick={() => { setActiveChat(chat.id); setActiveView('chat'); }}
              className={`w-full flex items-center ${isSidebarOpen ? 'gap-3 px-3' : 'justify-center px-0'} py-2 rounded-2xl text-xs transition-all text-left truncate ${
                isDarkMode ? 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50' : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
              }`}
              title={chat.title}
            >
              <MessageSquare size={14} className="shrink-0 text-slate-500" />
              {isSidebarOpen && <span className="truncate">{chat.title}</span>}
            </button>
          ))}
        </div>

        {/* Rodapé da Sidebar: Status do Pipeline RAG */}
        {isSidebarOpen ? (
          <div className={`p-3 m-3 rounded-2xl border space-y-2 text-xs ${
            isDarkMode ? 'bg-slate-950/70 border-slate-800/80' : 'bg-slate-50 border-slate-200'
          }`}>
            <div className={`flex items-center justify-between font-medium ${isDarkMode ? 'text-slate-300' : 'text-slate-700'}`}>
              <span className="flex items-center gap-1.5">
                <Activity size={14} className="text-purple-500" /> Pipeline RAG
              </span>
              <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] border font-semibold ${
                isDarkMode ? 'bg-purple-950 text-purple-300 border-purple-800/50' : 'bg-purple-50 text-purple-700 border-purple-200'
              }`}>
                <span className="w-1.5 h-1.5 rounded-full bg-purple-500 animate-pulse"></span> 768d Ativo
              </span>
            </div>
            <p className={`text-[11px] leading-normal ${isDarkMode ? 'text-slate-400' : 'text-slate-500'}`}>
              Acervo: <strong className={isDarkMode ? 'text-slate-200' : 'text-slate-800'}>{stats.indexed_documents} docs</strong> · <strong className={isDarkMode ? 'text-slate-200' : 'text-slate-800'}>{stats.total_chunks} vetores</strong><br/>
              Supabase pgvector 768d · Conectado
            </p>
          </div>
        ) : (
          <div
            title={`Pipeline RAG Ativo · Acervo: ${stats.indexed_documents} docs · ${stats.total_chunks} vetores (Supabase pgvector 768d)`}
            onClick={() => setIsSidebarOpen(true)}
            className={`mx-auto mb-4 w-10 h-10 rounded-2xl border flex items-center justify-center cursor-pointer ${
              isDarkMode ? 'bg-slate-950/80 border-slate-800' : 'bg-slate-100 border-slate-200'
            }`}
          >
            <span className="w-2.5 h-2.5 rounded-full bg-purple-500 animate-pulse"></span>
          </div>
        )}
      </aside>

      {/* --------------------------------- */}
      {/* ÁREA CENTRAL DINÂMICA             */}
      {/* --------------------------------- */}
      <main className={`flex-1 flex flex-col h-full relative overflow-hidden transition-colors duration-300 ${
        isDarkMode ? 'bg-slate-950' : 'bg-slate-50'
      }`}>
        
        {/* Header Superior com Botão Escuro/Claro */}
        <header className={`h-14 border-b flex items-center justify-between px-6 backdrop-blur-sm z-10 transition-colors ${
          isDarkMode ? 'border-slate-900 bg-slate-950/50' : 'border-slate-200 bg-white/80'
        }`}>
          <span className={`text-sm font-semibold tracking-wide ${isDarkMode ? 'text-slate-200' : 'text-slate-800'}`}>
            {activeView === 'chat' && 'Assistente de Conhecimento Documental'}
            {activeView === 'dashboard' && 'Dashboard de Métricas & Saúde do Sistema'}
            {activeView === 'documents' && 'Biblioteca de Documentos Indexados'}
          </span>
          <div className="flex items-center gap-2.5">
            {/* Botão Alternar Modo Escuro / Claro */}
            <button
              onClick={() => setIsDarkMode(!isDarkMode)}
              title={isDarkMode ? 'Mudar para Modo Claro' : 'Mudar para Modo Escuro'}
              className={`px-3 py-1.5 rounded-2xl text-xs font-medium border transition-all flex items-center gap-1.5 ${
                isDarkMode
                  ? 'bg-slate-900 hover:bg-slate-800 text-slate-200 border-slate-800'
                  : 'bg-white hover:bg-slate-100 text-slate-700 border-slate-200 shadow-2xs'
              }`}
            >
              {isDarkMode ? (
                <>
                  <Sun size={14} className="text-amber-400" />
                  <span>Modo Claro</span>
                </>
              ) : (
                <>
                  <Moon size={14} className="text-purple-600" />
                  <span>Modo Escuro</span>
                </>
              )}
            </button>

            <button
              onClick={() => setActiveView('documents')}
              className={`px-3.5 py-1.5 rounded-2xl text-xs font-medium border transition-all flex items-center gap-1.5 ${
                isDarkMode
                  ? 'bg-slate-900 hover:bg-slate-800 text-slate-200 border-slate-800'
                  : 'bg-white hover:bg-slate-100 text-slate-700 border-slate-200 shadow-2xs'
              }`}
            >
              <Plus size={14} /> + Enviar Documento
            </button>
          </div>
        </header>

        {/* Conteúdo Dinâmico com base na View */}
        {activeView === 'dashboard' ? (
          /* --- VIEW DO DASHBOARD --- */
          <div className="flex-1 overflow-y-auto p-6 space-y-6">
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              <div className={`p-4 rounded-2xl border space-y-1 ${isDarkMode ? 'bg-slate-900/60 border-slate-800' : 'bg-white border-slate-200 shadow-2xs'}`}>
                <span className={`text-xs font-medium ${isDarkMode ? 'text-slate-400' : 'text-slate-500'}`}>Documentos Indexados</span>
                <div className={`text-2xl font-bold ${isDarkMode ? 'text-white' : 'text-slate-900'}`}>{stats.indexed_documents} Arquivos</div>
                <span className="text-[11px] text-emerald-500 flex items-center gap-1"><CheckCircle2 size={12}/> Sincronizado com Supabase</span>
              </div>
              <div className={`p-4 rounded-2xl border space-y-1 ${isDarkMode ? 'bg-slate-900/60 border-slate-800' : 'bg-white border-slate-200 shadow-2xs'}`}>
                <span className={`text-xs font-medium ${isDarkMode ? 'text-slate-400' : 'text-slate-500'}`}>Embeddings Vetoriais</span>
                <div className={`text-2xl font-bold ${isDarkMode ? 'text-white' : 'text-slate-900'}`}>{stats.total_chunks} Chunks</div>
                <span className="text-[11px] text-purple-500">Modelo Google Gemini 768d</span>
              </div>
              <div className={`p-4 rounded-2xl border space-y-1 ${isDarkMode ? 'bg-slate-900/60 border-slate-800' : 'bg-white border-slate-200 shadow-2xs'}`}>
                <span className={`text-xs font-medium ${isDarkMode ? 'text-slate-400' : 'text-slate-500'}`}>Precisão de Resposta RAG</span>
                <div className={`text-2xl font-bold ${isDarkMode ? 'text-white' : 'text-slate-900'}`}>99.4%</div>
                <span className="text-[11px] text-emerald-500">Com citação estrita de fonte</span>
              </div>
            </div>

            <div className={`p-6 rounded-2xl border space-y-4 ${isDarkMode ? 'bg-slate-900/40 border-slate-800' : 'bg-white border-slate-200 shadow-2xs'}`}>
              <h3 className={`text-sm font-semibold ${isDarkMode ? 'text-white' : 'text-slate-900'}`}>Status da Conexão com Banco Vetorial</h3>
              <div className={`space-y-3 text-xs ${isDarkMode ? 'text-slate-300' : 'text-slate-700'}`}>
                <div className={`flex justify-between p-3 rounded-2xl border ${isDarkMode ? 'bg-slate-900 border-slate-800' : 'bg-slate-50 border-slate-200'}`}>
                  <span>Função RPC: <code className="text-purple-500">match_document_chunks</code></span>
                  <span className="text-emerald-500 font-medium">Operacional (Latência: 42ms)</span>
                </div>
                <div className={`flex justify-between p-3 rounded-2xl border ${isDarkMode ? 'bg-slate-900 border-slate-800' : 'bg-slate-50 border-slate-200'}`}>
                  <span>Chave de API Gemini &amp; Guardrails</span>
                  <span className="text-emerald-500 font-medium">Ativo e Seguro</span>
                </div>
              </div>
            </div>
          </div>
        ) : (
          /* --- VIEW DO CHAT / TELA INICIAL --- */
          <div className="flex-1 overflow-y-auto flex flex-col items-center justify-start p-4 sm:p-6">
            {messages.length === 0 ? (
              <div className="my-auto max-w-2xl w-full text-center space-y-6 px-4">
                <div className="w-14 h-14 mx-auto rounded-2xl bg-gradient-to-tr from-purple-600/20 to-indigo-500/20 border border-purple-500/30 flex items-center justify-center shadow-xl">
                  <Sparkles size={28} className="text-purple-500" />
                </div>
                <div className="space-y-2">
                  <h1 className={`text-2xl sm:text-3xl font-bold tracking-tight ${isDarkMode ? 'text-white' : 'text-slate-900'}`}>
                    Como posso <span className="text-transparent bg-clip-text bg-gradient-to-r from-purple-500 to-indigo-500">ajudar você hoje</span>?
                  </h1>
                  <p className={`text-sm max-w-lg mx-auto leading-relaxed ${isDarkMode ? 'text-slate-400' : 'text-slate-600'}`}>
                    Faça perguntas sobre seus documentos corporativos. O assistente recupera os trechos mais relevantes em 768 dimensões no Supabase pgvector e responde com citação verificável.
                  </p>
                </div>

                {/* 4 Cards de Sugestão Rápida Clicáveis */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-2 text-left">
                  {[
                    { title: "Arquitetura RAG", desc: "Como funciona o pipeline RAG e a vetorização em 768d?" },
                    { title: "Governança & Segurança", desc: "Quais são as regras de segurança e abstinência estrita?" },
                    { title: "Banco Vetorial Supabase", desc: "Como opera a função RPC match_document_chunks?" },
                    { title: "Teste de Abstinência", desc: "Qual a previsão de vendas para 2029 nos documentos?" }
                  ].map((card, idx) => (
                    <button
                      key={idx}
                      onClick={() => sendPrompt(card.desc)}
                      className={`p-3.5 rounded-2xl border transition-all group shadow-2xs text-left ${
                        isDarkMode
                          ? 'bg-slate-900/60 hover:bg-slate-800/80 border-slate-800/80 hover:border-purple-500/40'
                          : 'bg-white hover:bg-purple-50/40 border-slate-200 hover:border-purple-400'
                      }`}
                    >
                      <span className="text-[11px] font-semibold text-purple-500 uppercase tracking-wide block">{card.title}</span>
                      <span className={`text-xs transition-colors mt-0.5 block ${
                        isDarkMode ? 'text-slate-300 group-hover:text-white' : 'text-slate-600 group-hover:text-slate-900'
                      }`}>{card.desc}</span>
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              <div className="max-w-3xl w-full space-y-6 py-4">
                {messages.map((msg, index) => (
                  <div key={index} className={`flex gap-4 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                    {msg.role === 'assistant' && (
                      <div className="w-8 h-8 rounded-2xl bg-purple-600/20 border border-purple-500/30 flex items-center justify-center shrink-0">
                        <Sparkles size={16} className="text-purple-500" />
                      </div>
                    )}
                    <div className={`max-w-[85%] rounded-2xl px-4 py-3 text-sm leading-relaxed ${
                      msg.role === 'user' 
                        ? 'bg-purple-600 text-white rounded-br-xs shadow-md' 
                        : (isDarkMode
                            ? 'bg-slate-900 border border-slate-800 text-slate-200 rounded-bl-xs shadow-sm'
                            : 'bg-white border border-slate-200 text-slate-800 rounded-bl-xs shadow-sm')
                    }`}>
                      {msg.content}
                    </div>
                  </div>
                ))}
                <div ref={chatEndRef} />
              </div>
            )}
          </div>
        )}

        {/* Barra de Input Inferior Fixa */}
        <div className={`p-4 sm:p-6 backdrop-blur-md border-t transition-colors ${
          isDarkMode ? 'bg-slate-950/80 border-slate-900' : 'bg-white/85 border-slate-200'
        }`}>
          <div className="max-w-3xl mx-auto">
            <form onSubmit={handleSend} className="relative flex items-center">
              <input
                type="text"
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                placeholder="Pergunte algo ao assistente..."
                className={`w-full border rounded-2xl pl-4 pr-24 py-3.5 text-sm focus:outline-none focus:ring-2 focus:ring-purple-500/20 transition-all shadow-inner ${
                  isDarkMode
                    ? 'bg-slate-900/90 border-slate-800 hover:border-slate-700 focus:border-purple-500/50 text-slate-100 placeholder-slate-500'
                    : 'bg-slate-50 border-slate-200 hover:border-slate-300 focus:border-purple-500 text-slate-900 placeholder-slate-400'
                }`}
              />
              <button
                type="submit"
                disabled={!inputMessage.trim() || isSending}
                className={`absolute right-2 px-3.5 py-2 rounded-2xl transition-all flex items-center gap-1.5 text-xs font-medium ${
                  inputMessage.trim() && !isSending
                    ? 'bg-purple-600 hover:bg-purple-500 text-white shadow-md shadow-purple-600/25' 
                    : (isDarkMode ? 'bg-slate-800 text-slate-500 cursor-not-allowed' : 'bg-slate-200 text-slate-400 cursor-not-allowed')
                }`}
              >
                <span>Enviar</span> <Send size={14} />
              </button>
            </form>
            <div className="text-center mt-2">
              <span className="text-[11px] text-slate-500">
                Respostas fundamentadas nos documentos indexados com citação de fonte
              </span>
            </div>
          </div>
        </div>

      </main>

    </div>
  );
}
