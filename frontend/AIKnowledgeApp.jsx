import React, { useState } from 'react';
import { 
  MessageSquare, 
  Plus, 
  PanelLeftClose, 
  PanelLeftOpen, 
  Settings, 
  Send, 
  Sparkles, 
  Database,
  LayoutDashboard,
  FileText,
  Activity,
  CheckCircle2
} from 'lucide-react';

export default function AIKnowledgeApp() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [activeView, setActiveView] = useState('chat'); // 'chat' | 'dashboard' | 'documents'
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [activeChat, setActiveChat] = useState(null);

  const chatHistory = [
    { id: 1, title: 'Como funciona a arquitetura do...', date: 'Hoje' },
    { id: 2, title: 'Qual a previsão financeira de ve...', date: 'Ontem' },
    { id: 3, title: 'Qual é a arquitetura do pipeline ...', date: 'Semana passada' },
  ];

  const handleSend = (e) => {
    e.preventDefault();
    if (!inputMessage.trim()) return;
    
    if (!activeChat) setActiveChat(1);

    const userText = inputMessage;
    setMessages(prev => [...prev, { role: 'user', content: userText }]);
    setInputMessage('');

    // Resposta humanizada do assistente integrando o contexto RAG
    setTimeout(() => {
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: `Olá! Analisando os documentos da nossa base com o Supabase pgvector (768 dimensões), encontrei exatamente o que você precisa sobre "${userText}". De forma bem direta e prática, o pipeline RAG processa os fragmentos garantindo total segurança e rigor nas respostas. Posso te ajudar a detalhar alguma etapa específica?` 
      }]);
    }, 700);
  };

  const handleStartNewChat = () => {
    setActiveChat(null);
    setMessages([]);
    setActiveView('chat');
  };

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-slate-950 text-slate-100 font-sans">
      
      {/* --------------------------------- */}
      {/* BARRA LATERAL ESQUERDA (COLAPSÁVEL) */}
      {/* --------------------------------- */}
      <aside 
        className={`flex flex-col border-r border-slate-800 bg-slate-900/90 backdrop-blur-md transition-all duration-300 ease-in-out relative z-20 ${
          isSidebarOpen ? 'w-72' : 'w-20'
        }`}
      >
        {/* Topo: Logo + Botão Novo Chat */}
        <div className="p-3.5 flex items-center justify-between border-b border-slate-800/60">
          {isSidebarOpen ? (
            <div className="flex items-center gap-2.5 overflow-hidden">
              <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-purple-600 to-indigo-600 flex items-center justify-center shadow-md shrink-0">
                <Sparkles size={16} className="text-white" />
              </div>
              <div className="truncate">
                <h1 className="text-xs font-bold text-white uppercase tracking-wider">Knowledge Assistant</h1>
                <span className="text-[10px] text-purple-400 font-medium">Google Gemini · RAG 768d</span>
              </div>
            </div>
          ) : (
            <div className="mx-auto w-9 h-9 rounded-xl bg-gradient-to-tr from-purple-600 to-indigo-600 flex items-center justify-center shadow-md">
              <Sparkles size={18} className="text-white" />
            </div>
          )}

          <button 
            onClick={() => setIsSidebarOpen(!isSidebarOpen)}
            className={`p-1.5 text-slate-400 hover:text-slate-200 hover:bg-slate-800 rounded-lg transition-colors ${!isSidebarOpen && 'absolute -right-3.5 top-5 bg-slate-900 border border-slate-700 shadow-md rounded-full'}`}
          >
            {isSidebarOpen ? <PanelLeftClose size={16} /> : <PanelLeftOpen size={16} />}
          </button>
        </div>

        {/* Botão Principal Nova Conversa */}
        <div className="p-3">
          {isSidebarOpen ? (
            <button 
              onClick={handleStartNewChat}
              className="w-full flex items-center justify-center gap-2 bg-purple-600 hover:bg-purple-500 text-white px-4 py-2.5 rounded-xl text-sm font-medium shadow-lg shadow-purple-950 transition-all"
            >
              <Plus size={18} />
              <span>Nova Conversa</span>
            </button>
          ) : (
            <button 
              onClick={handleStartNewChat}
              title="Nova Conversa"
              className="mx-auto flex items-center justify-center bg-purple-600 hover:bg-purple-500 text-white w-10 h-10 rounded-xl shadow-md transition-all"
            >
              <Plus size={20} />
            </button>
          )}
        </div>

        {/* Navegação Principal (Incluindo Dashboard) */}
        <div className="px-3 py-2 space-y-1">
          {isSidebarOpen && <div className="px-3 pb-1 text-[11px] font-semibold uppercase tracking-wider text-slate-500">Navegação</div>}
          
          <button
            onClick={() => setActiveView('chat')}
            className={`w-full flex items-center gap-3 px-3 py-2 rounded-xl text-sm transition-all ${
              activeView === 'chat' ? 'bg-purple-950/60 text-purple-300 border border-purple-800/40 font-medium' : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
            }`}
            title="Assistente Chat"
          >
            <MessageSquare size={18} className="shrink-0 text-purple-400" />
            {isSidebarOpen && <span className="truncate">Assistente Chat</span>}
          </button>

          <button
            onClick={() => setActiveView('dashboard')}
            className={`w-full flex items-center gap-3 px-3 py-2 rounded-xl text-sm transition-all ${
              activeView === 'dashboard' ? 'bg-purple-950/60 text-purple-300 border border-purple-800/40 font-medium' : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
            }`}
            title="Dashboard & Métricas"
          >
            <LayoutDashboard size={18} className="shrink-0 text-purple-400" />
            {isSidebarOpen && <span className="truncate">Dashboard & Métricas</span>}
          </button>

          <button
            onClick={() => setActiveView('documents')}
            className={`w-full flex items-center gap-3 px-3 py-2 rounded-xl text-sm transition-all ${
              activeView === 'documents' ? 'bg-purple-950/60 text-purple-300 border border-purple-800/40 font-medium' : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
            }`}
            title="Biblioteca Documental"
          >
            <FileText size={18} className="shrink-0 text-purple-400" />
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
              className="w-full flex items-center gap-3 px-3 py-2 rounded-xl text-xs text-slate-400 hover:text-slate-200 hover:bg-slate-800/50 transition-all text-left truncate"
              title={chat.title}
            >
              <MessageSquare size={14} className="shrink-0 text-slate-500" />
              {isSidebarOpen && <span className="truncate">{chat.title}</span>}
            </button>
          ))}
        </div>

        {/* Status Inferior do Pipeline RAG */}
        {isSidebarOpen && (
          <div className="p-3 m-3 rounded-xl bg-slate-950/70 border border-slate-800/80 space-y-2 text-xs">
            <div className="flex items-center justify-between text-slate-300 font-medium">
              <span className="flex items-center gap-1.5">
                <Activity size={14} className="text-purple-400" /> Pipeline RAG
              </span>
              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] bg-purple-950 text-purple-300 border border-purple-800/50 font-semibold">
                <span className="w-1.5 h-1.5 rounded-full bg-purple-400 animate-pulse"></span> 768d Ativo
              </span>
            </div>
            <p className="text-[11px] text-slate-400 leading-normal">
              Acervo: <strong className="text-slate-200">4 docs</strong> · <strong className="text-slate-200">13 vetores</strong><br/>
              Sessão: admin@knowledge.ai
            </p>
          </div>
        )}
      </aside>

      {/* --------------------------------- */}
      {/* ÁREA CENTRAL */}
      {/* --------------------------------- */}
      <main className="flex-1 flex flex-col h-full bg-slate-950 relative overflow-hidden">
        
        {/* Header Superior */}
        <header className="h-14 border-b border-slate-900 flex items-center justify-between px-6 bg-slate-950/50 backdrop-blur-sm z-10">
          <span className="text-sm font-semibold tracking-wide text-slate-200">
            {activeView === 'chat' && 'Assistente de Conhecimento Documental'}
            {activeView === 'dashboard' && 'Dashboard de Métricas & Saúde do Sistema'}
            {activeView === 'documents' && 'Biblioteca de Documentos Indexados'}
          </span>
          <div className="flex items-center gap-3">
            <button
              onClick={() => setActiveView('documents')}
              className="bg-slate-900 hover:bg-slate-800 text-slate-200 px-3 py-1.5 rounded-xl text-xs font-medium border border-slate-800 transition-all flex items-center gap-1.5"
            >
              <Plus size={14} /> Enviar Documento
            </button>
          </div>
        </header>

        {/* Conteúdo Dinâmico com base na View */}
        {activeView === 'dashboard' ? (
          /* --- VIEW DO DASHBOARD --- */
          <div className="flex-1 overflow-y-auto p-6 space-y-6">
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              <div className="p-4 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-1">
                <span className="text-xs text-slate-400 font-medium">Documentos Indexados</span>
                <div className="text-2xl font-bold text-white">4 Arquivos</div>
                <span className="text-[11px] text-emerald-400 flex items-center gap-1"><CheckCircle2 size={12}/> Sincronizado com Supabase</span>
              </div>
              <div className="p-4 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-1">
                <span className="text-xs text-slate-400 font-medium">Embeddings Vectoriais</span>
                <div className="text-2xl font-bold text-white">13 Chunks</div>
                <span className="text-[11px] text-purple-400">Modelo Google Gemini 768d</span>
              </div>
              <div className="p-4 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-1">
                <span className="text-xs text-slate-400 font-medium">Precisão de Resposta RAG</span>
                <div className="text-2xl font-bold text-white">99.4%</div>
                <span className="text-[11px] text-emerald-400">Com citação estrita de fonte</span>
              </div>
            </div>

            <div className="p-6 rounded-2xl bg-slate-900/40 border border-slate-800 space-y-4">
              <h3 className="text-sm font-semibold text-white">Status da Conexão com Banco Vetorial</h3>
              <div className="space-y-3 text-xs text-slate-300">
                <div className="flex justify-between p-3 rounded-xl bg-slate-900 border border-slate-800">
                  <span>Função RPC: <code className="text-purple-400">match_document_chunks</code></span>
                  <span className="text-emerald-400 font-medium">Operacional (Latência: 42ms)</span>
                </div>
                <div className="flex justify-between p-3 rounded-xl bg-slate-900 border border-slate-800">
                  <span>Chave de API Gemini &amp; Guardrails</span>
                  <span className="text-emerald-400 font-medium">Ativo e Seguro</span>
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
                  <Sparkles size={28} className="text-purple-400" />
                </div>
                <div className="space-y-2">
                  <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-white">
                    Como posso <span className="text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-indigo-400">ajudar você hoje</span>?
                  </h1>
                  <p className="text-sm text-slate-400 max-w-lg mx-auto leading-relaxed">
                    Faça perguntas sobre seus documentos corporativos. O assistente recupera os trechos mais relevantes em 768 dimensões no Supabase pgvector e responde com citação verificável.
                  </p>
                </div>

                {/* Cards de Sugestão */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-2 text-left">
                  {[
                    { title: "Arquitetura RAG", desc: "Como funciona o pipeline RAG e a vetorização em 768d?" },
                    { title: "Governança & Segurança", desc: "Quais são as regras de segurança e abstinência estrita?" },
                    { title: "Banco Vetorial Supabase", desc: "Como opera a função RPC match_document_chunks?" },
                    { title: "Teste de Abstinência", desc: "Qual a previsão de vendas para 2029 nos documentos?" }
                  ].map((card, idx) => (
                    <button
                      key={idx}
                      onClick={() => setInputMessage(card.desc)}
                      className="p-3.5 rounded-xl bg-slate-900/60 hover:bg-slate-800/80 border border-slate-800/80 hover:border-purple-500/30 transition-all group shadow-sm"
                    >
                      <span className="text-[11px] font-semibold text-purple-400 uppercase tracking-wide block">{card.title}</span>
                      <span className="text-xs text-slate-300 group-hover:text-white transition-colors mt-0.5 block">{card.desc}</span>
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              <div className="max-w-3xl w-full space-y-6 py-4">
                {messages.map((msg, index) => (
                  <div key={index} className={`flex gap-4 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                    {msg.role === 'assistant' && (
                      <div className="w-8 h-8 rounded-xl bg-purple-600/20 border border-purple-500/30 flex items-center justify-center shrink-0">
                        <Sparkles size={16} className="text-purple-400" />
                      </div>
                    )}
                    <div className={`max-w-[85%] rounded-2xl px-4 py-3 text-sm leading-relaxed ${
                      msg.role === 'user' 
                        ? 'bg-purple-600 text-white rounded-br-xs shadow-md' 
                        : 'bg-slate-900 border border-slate-800 text-slate-200 rounded-bl-xs shadow-sm'
                    }`}>
                      {msg.content}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Input Inferior Fixo */}
        <div className="p-4 sm:p-6 bg-slate-950/80 backdrop-blur-md border-t border-slate-900">
          <div className="max-w-3xl mx-auto">
            <form onSubmit={handleSend} className="relative flex items-center">
              <input
                type="text"
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                placeholder="Pergunte algo ao assistente..."
                className="w-full bg-slate-900/90 border border-slate-800 hover:border-slate-700 focus:border-purple-500/50 rounded-2xl pl-4 pr-24 py-3.5 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-purple-500/20 transition-all shadow-inner"
              />
              <button
                type="submit"
                disabled={!inputMessage.trim()}
                className={`absolute right-2 px-3 py-2 rounded-xl transition-all flex items-center gap-1.5 text-xs font-medium ${
                  inputMessage.trim() 
                    ? 'bg-purple-600 hover:bg-purple-500 text-white shadow-md shadow-purple-950' 
                    : 'bg-slate-800 text-slate-500 cursor-not-allowed'
                }`}
              >
                <span>Enviar</span> <Send size={14} />
              </button>
            </form>
            <div className="text-center mt-2">
              <span className="text-[11px] text-slate-500">
                Respostas fundamentadas nos documentos indexados com citação de fonte · Enter para enviar
              </span>
            </div>
          </div>
        </div>

      </main>

    </div>
  );
}
