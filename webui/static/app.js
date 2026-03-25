/* Flash-MoE Chat — Vue 3 application */

const { createApp, ref, reactive, computed, watch, nextTick, onMounted } = Vue;

// ===== Marked configuration =====
marked.setOptions({ breaks: true, gfm: true });

const renderer = new marked.Renderer();
renderer.code = function (code, language) {
  var text, lang;
  if (typeof code === 'object' && code !== null) {
    text = code.text || '';
    lang = code.lang || '';
  } else {
    text = code || '';
    lang = language || '';
  }
  var langLabel = lang || 'code';
  var escaped = text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  var highlighted;
  if (lang && hljs.getLanguage(lang)) {
    try { highlighted = hljs.highlight(text, { language: lang }).value; } catch (e) { highlighted = escaped; }
  } else {
    highlighted = escaped;
  }
  return '<pre><div class="code-header"><span>' + langLabel + '</span>' +
    '<button class="copy-btn" onclick="window.__copyCode(this)">Copy</button></div>' +
    '<code class="hljs language-' + lang + '">' + highlighted + '</code></pre>';
};
marked.use({ renderer });

// ===== Global helpers =====
window.__copyCode = function (btn) {
  var code = btn.closest('pre').querySelector('code');
  navigator.clipboard.writeText(code.textContent).then(function () {
    btn.textContent = 'Copied!';
    setTimeout(function () { btn.textContent = 'Copy'; }, 2000);
  });
};

function generateId() {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 7);
}

function loadFromStorage(key, defaultVal) {
  try {
    var v = localStorage.getItem(key);
    return v ? JSON.parse(v) : defaultVal;
  } catch (e) { return defaultVal; }
}

// ===== Vue App =====
createApp({
  setup() {
    var sidebarOpen = ref(true);
    var showSettings = ref(false);
    var userInput = ref('');
    var isStreaming = ref(false);
    var isLoading = ref(false);
    var modelName = ref('');
    var connectionStatus = ref('checking');
    var connectionLabel = ref('Checking...');
    var messagesWrapper = ref(null);
    var inputEl = ref(null);
    var abortController = null;

    // Edit state
    var editingIdx = ref(-1);
    var editText = ref('');

    // --- Settings ---
    var defaultSettings = {
      baseUrl: 'http://localhost:8080',
      authHeader: '',
      model: '',
      systemPrompt: '',
      temperature: 0.7,
      topP: 1.0,
      maxTokens: 4096,
      minTokens: 0,
      minTokensEnabled: false,
      frequencyPenalty: 0.0,
      thinkingEnabled: true,
      cacheEnabled: true,
    };
    var settings = reactive(loadFromStorage('flashmoe-settings', Object.assign({}, defaultSettings)));
    Object.keys(defaultSettings).forEach(function (k) {
      if (settings[k] === undefined) settings[k] = defaultSettings[k];
    });

    // --- Chats ---
    var chats = reactive(loadFromStorage('flashmoe-chats', []));
    var activeChatId = ref(loadFromStorage('flashmoe-active-chat', null));

    var sortedChats = computed(function () {
      return chats.slice().sort(function (a, b) { return b.updatedAt - a.updatedAt; });
    });

    var activeChat = computed(function () {
      return chats.find(function (c) { return c.id === activeChatId.value; }) || null;
    });

    var displayMessages = computed(function () {
      if (!activeChat.value) return [];
      return activeChat.value.messages.filter(function (m) { return !m._prefill; });
    });

    // --- Persistence ---
    function saveChats() {
      localStorage.setItem('flashmoe-chats', JSON.stringify(chats));
      localStorage.setItem('flashmoe-active-chat', JSON.stringify(activeChatId.value));
    }
    function persistSettings() {
      localStorage.setItem('flashmoe-settings', JSON.stringify(Object.assign({}, settings)));
    }
    function saveSettings() {
      persistSettings();
      showSettings.value = false;
      checkHealth();
    }

    watch([function () { return chats.slice(); }, activeChatId], saveChats, { deep: true });

    // --- Invalidate cache session ---
    // Resets cached turn count so the server sends full context next time.
    // We use a separate sessionId for cache (not the chat.id) so that
    // invalidating cache does NOT break the activeChatId linkage.
    function invalidateCache(chat) {
      chat.cacheSessionId = generateId();
      chat.cachedTurnCount = 0;
    }

    // --- Chat management ---
    function newChat() {
      var chat = {
        id: generateId(),
        title: 'New Chat',
        messages: [],
        createdAt: Date.now(),
        updatedAt: Date.now(),
        cachedTurnCount: 0,
        cacheSessionId: generateId(),
      };
      chats.push(chat);
      activeChatId.value = chat.id;
      nextTick(function () { if (inputEl.value) inputEl.value.focus(); });
    }

    function selectChat(id) {
      activeChatId.value = id;
      editingIdx.value = -1;
      nextTick(function () { scrollToBottom(); if (inputEl.value) inputEl.value.focus(); });
    }

    function deleteChat(id) {
      var idx = chats.findIndex(function (c) { return c.id === id; });
      if (idx !== -1) {
        chats.splice(idx, 1);
        if (activeChatId.value === id) {
          activeChatId.value = chats.length > 0 ? chats[chats.length - 1].id : null;
        }
      }
    }

    function scrollToBottom() {
      nextTick(function () {
        if (messagesWrapper.value) {
          messagesWrapper.value.scrollTop = messagesWrapper.value.scrollHeight;
        }
      });
    }

    function autoResize() {
      var el = inputEl.value;
      if (el) { el.style.height = 'auto'; el.style.height = Math.min(el.scrollHeight, 200) + 'px'; }
    }

    function handleKeydown(e) {
      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
    }

    // --- Think helpers ---
    function isThinkingOnly(text) {
      if (!text) return true;
      var openCount = (text.match(/<think>/g) || []).length;
      var closeCount = (text.match(/<\/think>/g) || []).length;
      if (openCount > closeCount) return true;
      return text.replace(/<think>[\s\S]*?<\/think>/g, '').trim() === '';
    }

    // --- Markdown rendering ---
    function renderMarkdown(text) {
      if (!text) return '';
      var result = '';
      var remaining = text;
      while (true) {
        var openPos = remaining.indexOf('<think>');
        if (openPos === -1) break;
        var before = remaining.slice(0, openPos);
        if (before.trim()) result += marked.parse(before);
        var afterOpen = remaining.slice(openPos + 7);
        var closePos = afterOpen.indexOf('</think>');
        if (closePos === -1) {
          var thinkContent = afterOpen;
          var rendered = thinkContent.trim() ? marked.parse(thinkContent.trim()) : '';
          result += '<details class="think-block think-streaming" open>' +
            '<summary class="think-header"><span class="think-label">Thinking</span></summary>' +
            '<div class="think-body">' + rendered + '</div></details>';
          remaining = '';
          break;
        } else {
          var thinkContent = afterOpen.slice(0, closePos);
          remaining = afterOpen.slice(closePos + 8);
          if (thinkContent.trim()) {
            var rendered = marked.parse(thinkContent.trim());
            result += '<details class="think-block" open>' +
              '<summary class="think-header"><span class="think-label">Thought process</span></summary>' +
              '<div class="think-body">' + rendered + '</div></details>';
          }
        }
      }
      if (remaining.trim()) result += marked.parse(remaining);
      return result;
    }

    // --- Edit messages ---
    function startEdit(idx) {
      var msgs = displayMessages.value;
      editingIdx.value = idx;
      editText.value = msgs[idx].content;
    }

    function cancelEdit() {
      editingIdx.value = -1;
      editText.value = '';
    }

    function saveEdit(idx) {
      var chat = activeChat.value;
      if (!chat) return;
      var msgs = displayMessages.value;
      var msg = msgs[idx];

      // Find real index in chat.messages
      var realIdx = -1;
      var count = 0;
      for (var i = 0; i < chat.messages.length; i++) {
        if (chat.messages[i]._prefill) continue;
        if (count === idx) { realIdx = i; break; }
        count++;
      }
      if (realIdx === -1) return;

      // Update message content
      chat.messages[realIdx].content = editText.value;

      // Truncate everything after this message (conversation branches)
      chat.messages.splice(realIdx + 1);

      // Invalidate cache — conversation changed
      invalidateCache(chat);

      chat.updatedAt = Date.now();
      editingIdx.value = -1;
      editText.value = '';

      // If edited message was user, auto-send to get new response
      if (msg.role === 'user') {
        resendFromHistory();
      }
    }

    // --- Delete a single message ---
    function deleteMessage(idx) {
      var chat = activeChat.value;
      if (!chat) return;
      var realIdx = -1;
      var count = 0;
      for (var i = 0; i < chat.messages.length; i++) {
        if (chat.messages[i]._prefill) continue;
        if (count === idx) { realIdx = i; break; }
        count++;
      }
      if (realIdx === -1) return;
      // Remove this message and everything after it
      chat.messages.splice(realIdx);
      invalidateCache(chat);
      chat.updatedAt = Date.now();
      saveChats();
      scrollToBottom();
    }

    // --- Regenerate assistant response ---
    function regenerate(idx) {
      var chat = activeChat.value;
      if (!chat || isStreaming.value) return;
      var msgs = displayMessages.value;
      if (msgs[idx].role !== 'assistant') return;

      // Find real index
      var realIdx = -1;
      var count = 0;
      for (var i = 0; i < chat.messages.length; i++) {
        if (chat.messages[i]._prefill) continue;
        if (count === idx) { realIdx = i; break; }
        count++;
      }
      if (realIdx === -1) return;

      // Remove this assistant message and everything after
      chat.messages.splice(realIdx);
      invalidateCache(chat);
      chat.updatedAt = Date.now();

      // Re-send to get new response
      resendFromHistory();
    }

    // --- Continue assistant response (prefill with existing content) ---
    function continueResponse(idx) {
      var chat = activeChat.value;
      if (!chat || isStreaming.value) return;
      var msgs = displayMessages.value;
      var msg = msgs[idx];
      if (msg.role !== 'assistant' || !msg.content) return;

      // Find real index
      var realIdx = -1;
      var count = 0;
      for (var i = 0; i < chat.messages.length; i++) {
        if (chat.messages[i]._prefill) continue;
        if (count === idx) { realIdx = i; break; }
        count++;
      }
      if (realIdx === -1) return;

      // Remove everything after this message
      chat.messages.splice(realIdx + 1);
      invalidateCache(chat);

      // Use the existing content as prefill and stream more
      var existingContent = msg.content;
      requestCompletion(chat, realIdx, existingContent);
    }

    // --- Resend from current history (after edit/regen) ---
    function resendFromHistory() {
      var chat = activeChat.value;
      if (!chat) return;

      // Add assistant placeholder
      chat.messages.push({ role: 'assistant', content: '' });
      var assistantIdx = chat.messages.length - 1;

      requestCompletion(chat, assistantIdx, null);
    }

    // --- Core completion request ---
    // prefillContent: if set, the assistant message already has this content
    // and we want the model to continue from it (not replace it).
    async function requestCompletion(chat, assistantIdx, prefillContent) {
      isLoading.value = true;
      isStreaming.value = true;

      var isContinuation = settings.cacheEnabled && (chat.cachedTurnCount || 0) > 0;
      var apiMessages = [];

      if (isContinuation) {
        // Find last user message
        var lastUserContent = '';
        for (var i = chat.messages.length - 1; i >= 0; i--) {
          if (chat.messages[i].role === 'user') {
            lastUserContent = chat.messages[i].content;
            break;
          }
        }
        apiMessages.push({ role: 'user', content: lastUserContent });
      } else {
        if (settings.systemPrompt) {
          apiMessages.push({ role: 'system', content: settings.systemPrompt });
        }
        for (var i = 0; i < chat.messages.length; i++) {
          var m = chat.messages[i];
          if (m._prefill) continue;
          if (m.role === 'user' || (m.role === 'assistant' && m.content)) {
            apiMessages.push({ role: m.role, content: m.content });
          }
        }
        // Remove trailing empty assistant
        if (apiMessages.length > 0 &&
            apiMessages[apiMessages.length - 1].role === 'assistant' &&
            !apiMessages[apiMessages.length - 1].content) {
          apiMessages.pop();
        }
      }

      // Prefill: either for continue (existing content) or thinking disabled
      if (prefillContent) {
        // Continue mode: prepend think-skip if needed, then existing content
        var prefill = '';
        if (!settings.thinkingEnabled) prefill += '<think>\n\n</think>\n';
        prefill += prefillContent;
        apiMessages.push({ role: 'assistant', content: prefill });
      } else if (!settings.thinkingEnabled) {
        apiMessages.push({ role: 'assistant', content: '<think>\n\n</think>\n' });
      }

      var reqBody = {
        base_url: settings.baseUrl,
        auth_header: settings.authHeader,
        messages: apiMessages,
        temperature: settings.temperature,
        top_p: settings.topP,
        max_tokens: settings.maxTokens,
        frequency_penalty: settings.frequencyPenalty,
        stream: true,
        stream_options: { include_usage: true },
        cache: settings.cacheEnabled,
      };
      if (settings.minTokensEnabled && settings.minTokens > 0) {
        reqBody.min_tokens = settings.minTokens;
      }
      if (settings.model) reqBody.model = settings.model;
      if (settings.cacheEnabled) reqBody.session_id = chat.cacheSessionId || chat.id;

      abortController = new AbortController();

      try {
        var resp = await fetch('/api/chat/stream', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(reqBody),
          signal: abortController.signal,
        });

        if (!resp.ok) {
          var err = await resp.json().catch(function () { return { error: { message: 'Unknown error' } }; });
          chat.messages[assistantIdx] = { role: 'assistant', content: '', error: (err.error && err.error.message) || 'Request failed' };
          return;
        }

        // For continue mode, start with existing content
        if (prefillContent) {
          chat.messages[assistantIdx].content = prefillContent;
        }

        var reader = resp.body.getReader();
        var decoder = new TextDecoder();
        var buffer = '';

        while (true) {
          var readResult = await reader.read();
          if (readResult.done) break;
          buffer += decoder.decode(readResult.value, { stream: true });
          var lines = buffer.split('\n');
          buffer = lines.pop() || '';
          for (var li = 0; li < lines.length; li++) {
            var trimmed = lines[li].trim();
            if (!trimmed || !trimmed.startsWith('data:')) continue;
            var data = trimmed.slice(5).trim();
            if (data === '[DONE]') continue;
            try {
              var parsed = JSON.parse(data);
              if (parsed.error) {
                chat.messages[assistantIdx] = { role: 'assistant', content: '', error: parsed.error.message || 'Stream error' };
                return;
              }
              var delta = parsed.choices && parsed.choices[0] && parsed.choices[0].delta;
              if (delta && delta.content) {
                chat.messages[assistantIdx].content += delta.content;
                scrollToBottom();
              }
              if (parsed.usage) {
                chat.messages[assistantIdx].usage = parsed.usage;
              }
            } catch (e) { /* skip */ }
          }
        }
      } catch (e) {
        if (e.name !== 'AbortError') {
          chat.messages[assistantIdx] = {
            role: 'assistant', content: prefillContent || '',
            error: 'Connection error: ' + e.message,
          };
        }
      } finally {
        isLoading.value = false;
        isStreaming.value = false;
        abortController = null;
        chat.updatedAt = Date.now();
        if (settings.cacheEnabled && chat.messages[assistantIdx] &&
            chat.messages[assistantIdx].content && !chat.messages[assistantIdx].error) {
          chat.cachedTurnCount = (chat.cachedTurnCount || 0) + 1;
        }
        scrollToBottom();
      }
    }

    // --- Send message (main entry point) ---
    async function sendMessage() {
      var text = userInput.value.trim();
      if (!text || isLoading.value || isStreaming.value) return;

      if (!activeChat.value) { newChat(); await nextTick(); }
      var chat = activeChat.value;

      // Remove trailing empty assistant message (leftover from prefill abort)
      while (chat.messages.length > 0 &&
             chat.messages[chat.messages.length - 1].role === 'assistant' &&
             !chat.messages[chat.messages.length - 1].content) {
        chat.messages.pop();
      }

      chat.messages.push({ role: 'user', content: text });
      if (chat.title === 'New Chat') {
        chat.title = text.slice(0, 50) + (text.length > 50 ? '\u2026' : '');
      }
      chat.updatedAt = Date.now();
      userInput.value = '';
      nextTick(function () { autoResize(); scrollToBottom(); });

      chat.messages.push({ role: 'assistant', content: '' });
      var assistantIdx = chat.messages.length - 1;

      requestCompletion(chat, assistantIdx, null);
    }

    function stopStreaming() {
      if (abortController) { abortController.abort(); abortController = null; }
    }

    // --- Health check ---
    async function checkHealth() {
      connectionStatus.value = 'checking';
      connectionLabel.value = 'Checking...';
      try {
        var params = new URLSearchParams({ base_url: settings.baseUrl });
        if (settings.authHeader) params.set('auth_header', settings.authHeader);
        var resp = await fetch('/api/health?' + params);
        var data = await resp.json();
        if (data.status === 'ok' || data.model) {
          connectionStatus.value = 'connected';
          modelName.value = data.model || '';
          connectionLabel.value = 'Connected' + (data.model ? ' \u2014 ' + data.model : '');
        } else if (data.status === 'disconnected' || data.error) {
          connectionStatus.value = 'disconnected';
          connectionLabel.value = data.error || 'Disconnected';
          modelName.value = '';
        } else {
          connectionStatus.value = 'connected';
          connectionLabel.value = 'Connected';
        }
      } catch (e) {
        connectionStatus.value = 'disconnected';
        connectionLabel.value = 'WebUI server error';
        modelName.value = '';
      }
    }

    onMounted(function () {
      checkHealth();
      if (inputEl.value) inputEl.value.focus();
      setInterval(checkHealth, 30000);
    });

    return {
      sidebarOpen: sidebarOpen, showSettings: showSettings, userInput: userInput,
      isStreaming: isStreaming, isLoading: isLoading,
      modelName: modelName, connectionStatus: connectionStatus, connectionLabel: connectionLabel,
      messagesWrapper: messagesWrapper, inputEl: inputEl,
      settings: settings, chats: chats, activeChatId: activeChatId,
      sortedChats: sortedChats, activeChat: activeChat, displayMessages: displayMessages,
      editingIdx: editingIdx, editText: editText,
      newChat: newChat, selectChat: selectChat, deleteChat: deleteChat,
      sendMessage: sendMessage, stopStreaming: stopStreaming,
      handleKeydown: handleKeydown, autoResize: autoResize,
      renderMarkdown: renderMarkdown, isThinkingOnly: isThinkingOnly,
      saveSettings: saveSettings, persistSettings: persistSettings, checkHealth: checkHealth,
      startEdit: startEdit, cancelEdit: cancelEdit, saveEdit: saveEdit,
      deleteMessage: deleteMessage, regenerate: regenerate, continueResponse: continueResponse,
    };
  }
}).mount('#app');
