(() => {
    "use strict";

    const state = {
        documents: [],
        isProcessing: false,
        isStreaming: false,
        enabled: false,
    };

    const els = {
        sidebar: document.getElementById("sidebar"),
        themeToggle: document.getElementById("themeToggle"),
        uploadCard: document.getElementById("uploadCard"),
        fileInput: document.getElementById("fileInput"),
        documents: document.getElementById("documents"),
        docsEmpty: document.getElementById("docsEmpty"),
        chatMessages: document.getElementById("chatMessages"),
        emptyState: document.getElementById("emptyState"),
        suggestions: document.getElementById("suggestions"),
        chatInput: document.getElementById("chatInput"),
        sendBtn: document.getElementById("sendBtn"),
        newChat: document.getElementById("newChat"),
        statusText: document.getElementById("statusText"),
        toast: document.getElementById("toast"),
        dropOverlay: document.getElementById("dropOverlay"),
        composer: document.getElementById("composer"),
    };

    /* ---------- Theme ---------- */

    const THEME_KEY = "rag-theme";

    function applyTheme(theme) {
        document.documentElement.setAttribute("data-theme", theme);
        localStorage.setItem(THEME_KEY, theme);
    }

    function initTheme() {
        const saved = localStorage.getItem(THEME_KEY);
        if (saved === "light" || saved === "dark") {
            applyTheme(saved);
            return;
        }
        const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
        applyTheme(prefersDark ? "dark" : "light");
    }

    els.themeToggle.addEventListener("click", () => {
        const current = document.documentElement.getAttribute("data-theme");
        applyTheme(current === "dark" ? "light" : "dark");
    });

    /* ---------- Helpers ---------- */

    function toast(message, type = "") {
        els.toast.textContent = message;
        els.toast.className = "toast show" + (type ? " " + type : "");
        clearTimeout(toast._timer);
        toast._timer = setTimeout(() => {
            els.toast.className = "toast";
        }, 3200);
    }

    function escapeHtml(text) {
        const div = document.createElement("div");
        div.textContent = text;
        return div.innerHTML;
    }

    function formatBytes(bytes) {
        if (bytes < 1024) return bytes + " B";
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
        return (bytes / (1024 * 1024)).toFixed(1) + " MB";
    }

    /* ---------- Upload ---------- */

    function uploadFiles(files) {
        const pdfs = Array.from(files).filter(
            (f) => f.type === "application/pdf" || f.name.toLowerCase().endsWith(".pdf")
        );

        if (pdfs.length === 0) {
            toast("Please select PDF files only.", "error");
            return;
        }

        state.isProcessing = true;
        setUIState();

        pdfs.forEach((file) => addDocumentItem(file.name, "processing", "Processing..."));

        const uploads = pdfs.map((file) => {
            const form = new FormData();
            form.append("file", file);
            return fetch("/api/upload", { method: "POST", body: form })
                .then(async (res) => {
                    const data = await res.json().catch(() => ({}));
                    if (!res.ok) throw new Error(data.detail || "Upload failed");
                    return pollUpload(file, data.job_id);
                });
        });

        Promise.allSettled(uploads).then((results) => {
            state.isProcessing = false;

            results.forEach((result, i) => {
                const file = pdfs[i];
                if (result.status === "fulfilled") {
                    const data = result.value.data;
                    if (data.status === "duplicate") {
                        updateDocumentItem(file.name, "done", "Already uploaded — skipped");
                        state.documents.push(file.name);
                        toast(`"${file.name}" is a duplicate, skipped.`);
                    } else {
                        updateDocumentItem(
                            file.name,
                            "done",
                            `${data.chunks} chunks · ${data.pages} pages`
                        );
                        state.documents.push(file.name);
                        toast(`Uploaded "${file.name}" successfully.`);
                    }
                } else {
                    updateDocumentItem(file.name, "error", "Failed");
                    toast(`Failed to upload "${file.name}": ${result.reason.message}`, "error");
                }
            });

            if (state.documents.length > 0) {
                setUIState();
            }
            updateDocsEmpty();
            enableChat(state.documents.length > 0);
        });
    }

    async function pollUpload(file, jobId) {
        let delay = 1000;
        while (true) {
            await new Promise((r) => setTimeout(r, delay));
            const res = await fetch(`/api/upload/${jobId}`);
            if (!res.ok) throw new Error("Upload status check failed");
            const data = await res.json();

            if (data.status === "processing") {
                updateDocumentItem(file.name, "processing", data.message || "Processing...");
                delay = Math.min(delay + 500, 3000);
                continue;
            }
            if (data.status === "error") throw new Error(data.message || "Upload failed");
            if (data.status === "duplicate") return { file, data };
            if (data.status === "done") return { file, data };
            throw new Error("Unknown upload status");
        }
    }

    els.uploadCard.addEventListener("click", () => els.fileInput.click());
    els.fileInput.addEventListener("change", () => {
        uploadFiles(els.fileInput.files);
        els.fileInput.value = "";
    });

    /* ---------- Drag & drop ---------- */

    let dragDepth = 0;

    window.addEventListener("dragenter", (e) => {
        e.preventDefault();
        if (e.dataTransfer && e.dataTransfer.types.includes("Files")) {
            dragDepth++;
            els.dropOverlay.classList.add("active");
        }
    });

    window.addEventListener("dragover", (e) => e.preventDefault());

    window.addEventListener("dragleave", (e) => {
        e.preventDefault();
        dragDepth--;
        if (dragDepth <= 0) {
            dragDepth = 0;
            els.dropOverlay.classList.remove("active");
        }
    });

    window.addEventListener("drop", (e) => {
        e.preventDefault();
        dragDepth = 0;
        els.dropOverlay.classList.remove("active");
        if (e.dataTransfer.files.length) uploadFiles(e.dataTransfer.files);
    });

    els.uploadCard.addEventListener("dragover", (e) => {
        e.preventDefault();
        els.uploadCard.classList.add("dragover");
    });

    els.uploadCard.addEventListener("dragleave", () => {
        els.uploadCard.classList.remove("dragover");
    });

    /* ---------- Documents ---------- */

    function addDocumentItem(name, status, meta) {
        els.docsEmpty.style.display = "none";
        const item = document.createElement("div");
        item.className = "doc-item";
        item.style.animationDelay = Math.min(els.documents.children.length * 60, 420) + "ms";
        item.id = "doc-" + name.replace(/[^a-zA-Z0-9]/g, "-");
        item.innerHTML = `
            <div class="doc-icon">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                    <path d="M14 2v6h6" />
                </svg>
            </div>
            <div class="doc-info">
                <div class="doc-name">${escapeHtml(name)}</div>
                <div class="doc-meta">${escapeHtml(meta)}</div>
            </div>
            <div class="doc-status ${status === "processing" ? "processing" : ""}" ${
            status === "error" ? 'style="background:var(--red);box-shadow:0 0 8px var(--red);"' : ""
        }></div>
        `;
        els.documents.appendChild(item);
        return item;
    }

    function updateDocumentItem(name, status, meta) {
        const item = document.getElementById("doc-" + name.replace(/[^a-zA-Z0-9]/g, "-"));
        if (!item) return;
        const dot = item.querySelector(".doc-status");
        const metaEl = item.querySelector(".doc-meta");

        dot.className = "doc-status";
        dot.removeAttribute("style");
        if (status === "processing") {
            dot.classList.add("processing");
        } else if (status === "error") {
            dot.style.background = "var(--red)";
            dot.style.boxShadow = "0 0 8px var(--red)";
        }
        metaEl.textContent = meta;
    }

    function updateDocsEmpty() {
        els.docsEmpty.style.display = state.documents.length ? "none" : "block";
    }

    function setUIState() {
        els.uploadCard.classList.toggle("loading", state.isProcessing);
        els.uploadCard.querySelector("p").textContent = state.isProcessing
            ? "Processing uploads..."
            : "Drop your documents here or click to browse";
    }

    function enableChat(enable) {
        state.enabled = enable;
        els.chatInput.disabled = !enable;
        els.sendBtn.disabled = !enable;
        els.statusText.textContent = enable
            ? "Ready — ask about your documents"
            : "Upload a PDF to get started";
    }

    /* ---------- Suggestion chips ---------- */

    els.suggestions.addEventListener("click", (e) => {
        const chip = e.target.closest(".suggestion-chip");
        if (!chip || !state.enabled) return;
        els.chatInput.value = chip.querySelector("span").textContent;
        autoResize();
        sendQuestion();
    });

    /* ---------- Chat ---------- */

    function addMessage(role, content, sources = []) {
        const message = document.createElement("div");
        message.className = "message " + role;

        if (role === "user") {
            message.innerHTML = `
                <div class="avatar">You</div>
                <div class="bubble">${escapeHtml(content)}</div>
            `;
        } else {
            const md = renderMarkdown(content);
            const sourcesHtml = sources.length
                ? `<div class="sources">${sources
                      .map(
                          (s) =>
                              `<span class="source-chip">
                                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                                    <path d="M14 2v6h6" />
                                </svg>
                                ${escapeHtml(
                                    String(s.source).split("\\").pop().split("/").pop()
                                )} · p.${s.page + 1}
                              </span>`
                      )
                      .join("")}
                  </div>`
                : "";
            message.innerHTML = `
                <div class="avatar">AI</div>
                <div class="bubble">
                    <div class="md-body">${md}</div>
                    ${sourcesHtml}
                </div>
            `;
        }

        message.style.animationDelay = Math.min(els.chatMessages.children.length * 40, 400) + "ms";
        els.chatMessages.appendChild(message);
        scrollToBottom();
        return message;
    }

    function addTypingIndicator() {
        const message = document.createElement("div");
        message.className = "message assistant";
        message.id = "typing";
        message.innerHTML = `
            <div class="avatar">AI</div>
            <div class="bubble"><div class="typing"><span></span><span></span><span></span></div></div>
        `;
        els.chatMessages.appendChild(message);
        scrollToBottom();
    }

    function removeTypingIndicator() {
        const typing = document.getElementById("typing");
        if (typing) typing.remove();
    }

    function scrollToBottom() {
        els.chatMessages.scrollTop = els.chatMessages.scrollHeight;
    }

    async function sendQuestion() {
        const question = els.chatInput.value.trim();
        if (!question || state.isStreaming || !state.enabled) return;

        state.isStreaming = true;
        els.chatInput.value = "";
        autoResize();
        els.sendBtn.disabled = true;
        els.chatInput.disabled = true;
        hideEmptyState();

        addMessage("user", question);
        addTypingIndicator();

        try {
            const res = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ question }),
            });

            const data = await res.json().catch(() => ({}));
            if (!res.ok) throw new Error(data.detail || "Something went wrong");

            removeTypingIndicator();
            addMessage("assistant", data.answer || "No answer returned.", data.sources || []);
        } catch (err) {
            removeTypingIndicator();
            addMessage("assistant", `Error: ${err.message}`);
            toast(err.message, "error");
        } finally {
            state.isStreaming = false;
            els.chatInput.disabled = !state.enabled;
            els.sendBtn.disabled = !state.enabled;
            els.chatInput.focus();
        }
    }

    function hideEmptyState() {
        els.emptyState.classList.add("hidden");
    }

    function showEmptyState() {
        els.emptyState.classList.remove("hidden");
    }

    els.sendBtn.addEventListener("click", sendQuestion);
    els.chatInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendQuestion();
        }
    });

    /* ---------- New chat ---------- */

    els.newChat.addEventListener("click", () => {
        els.chatMessages.innerHTML = "";
        showEmptyState();
        toast("Chat cleared.");
    });

    /* ---------- Composer autosize ---------- */

    function autoResize() {
        els.chatInput.style.height = "auto";
        els.chatInput.style.height = Math.min(els.chatInput.scrollHeight, 140) + "px";
    }

    els.chatInput.addEventListener("input", autoResize);

    /* ---------- Tiny markdown renderer ---------- */

    function renderMarkdown(text) {
        let html = escapeHtml(text);

        html = html.replace(/^### (.*)$/gm, "<h3>$1</h3>");
        html = html.replace(/^## (.*)$/gm, "<h2>$1</h2>");
        html = html.replace(/^# (.*)$/gm, "<h1>$1</h1>");
        html = html.replace(/^&gt; (.*)$/gm, "<blockquote>$1</blockquote>");
        html = html.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
        html = html.replace(/\*(.*?)\*/g, "<em>$1</em>");
        html = html.replace(/`([^`\n]+)`/g, "<code>$1</code>");
        html = html.replace(/```([\s\S]*?)```/g, (m, code) => "<pre>" + code + "</pre>");
        html = html.replace(/(?:^|[^\S\n])(https?:\/\/[^\s<]+)/g, (m, url) =>
            `<a href="${url}" target="_blank" rel="noopener noreferrer">${url}</a>`
        );

        html = html.split(/\n{2,}/).map((block) => {
            if (/^<(h1|h2|h3|pre|blockquote|ul|ol)/.test(block.trim())) return block.trim();

            if (/^- /.test(block) || /^\* /.test(block)) {
                const items = block
                    .split(/\n/)
                    .map((l) => l.replace(/^[-*] /, ""))
                    .filter(Boolean)
                    .map((l) => "<li>" + l + "</li>")
                    .join("");
                return "<ul>" + items + "</ul>";
            }

            if (/^\d+\. /.test(block)) {
                const items = block
                    .split(/\n/)
                    .map((l) => l.replace(/^\d+\. /, ""))
                    .filter(Boolean)
                    .map((l) => "<li>" + l + "</li>")
                    .join("");
                return "<ol>" + items + "</ol>";
            }

            return "<p>" + block.trim() + "</p>";
        }).join("\n");

        return html;
    }

    initTheme();
    enableChat(false);
    els.chatInput.focus();
})();
