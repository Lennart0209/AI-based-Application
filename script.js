// Hauptklasse für die ganze App
class PDFChatApp {
  constructor() {
    // Hier wird alles gespeichert, was gerade aktiv ist
    this.currentChatId = null;
    this.chats = [];
    this.currentPDF = null;
    this.secondPDF = null;
    this.source = "1"; // Standardmäßig PDF 1 aktiv

    // Alle wichtigen HTML-Elemente zwischenspeichern
    this.elems = {
      uploadArea1: document.getElementById("uploadArea"),
      fileInput1: document.getElementById("fileInput"),
      uploadedFile1: document.getElementById("uploadedFile"),
      fileName1: document.getElementById("fileName"),

      uploadArea2: document.getElementById("uploadArea2"),
      fileInput2: document.getElementById("fileInput2"),
      uploadedFile2: document.getElementById("uploadedFile2"),
      fileName2: document.getElementById("fileName2"),

      chatMessages: document.getElementById("chatMessages"),
      chatInput: document.getElementById("chatInput"),
      sendButton: document.getElementById("sendButton"),
      newChatBtn: document.getElementById("newChatBtn"),
      chatList: document.getElementById("chatList"),

      sourceRadios: document.querySelectorAll('input[name="sourceSelect"]'),
      previewSidebar: document.querySelector(".preview-sidebar"),

      exportChatBtn: document.getElementById("exportChatBtn"),
      exportKeypointsBtn: document.getElementById("exportKeypointsBtn"),
    };

    // Events aktivieren und direkt neuen Chat starten
    this.bindEvents();
    this.createNewChat();
  }

  bindEvents() {
    // Für beide PDF Upload Bereiche Events setzen
    [1, 2].forEach((num) => this.setupUpload(num));

    // Nachricht abschicken beim Klick auf den Button
    this.elems.sendButton.addEventListener("click", () => this.sendMessage());

    // Enter-Taste abschicken (ohne Shift)
    this.elems.chatInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        this.sendMessage();
      }
    });

    // Neuer Chat Button
    this.elems.newChatBtn.addEventListener("click", () => this.createNewChat());

    // Quelle ändern (PDF 1 / PDF 2 / Beide)
    this.elems.sourceRadios.forEach((r) =>
      r.addEventListener("change", (e) => (this.source = e.target.value))
    );

    // Export Buttons
    if (this.elems.exportChatBtn)
      this.elems.exportChatBtn.addEventListener("click", () =>
        this.exportCurrentChatAsJSON()
      );
    if (this.elems.exportKeypointsBtn)
      this.elems.exportKeypointsBtn.addEventListener("click", () =>
        this.exportKeypoints()
      );
  }

  // Datei-Upload vorbereiten
  setupUpload(num) {
    const uploadArea = this.elems[`uploadArea${num}`];
    const fileInput = this.elems[`fileInput${num}`];

    // Beim Klicken auf das Upload-Feld die Dateiauswahl öffnen
    uploadArea.onclick = () => fileInput.click();

    // Datei wurde ausgewählt
    fileInput.onchange = (e) => {
      const file = e.target.files[0];
      if (file?.type === "application/pdf") this.handleFile(num, file);
    };
  }

  // Upload verarbeiten
  async handleFile(num, file) {
    const formData = new FormData();
    formData.append("file", file);

    try {
      const url = num === 1 ? "/upload" : "/upload2";
      const res = await fetch(url, { method: "POST", body: formData });
      if (!res.ok) throw new Error();

      // Dateiname anzeigen, Upload-Bereich verstecken
      this.elems[`fileName${num}`].textContent = file.name;
      this.elems[`uploadArea${num}`].style.display = "none";
      this.elems[`uploadedFile${num}`].style.display = "flex";

      if (num === 1) {
        this.currentPDF = file.name;
        this.elems.chatInput.disabled = false;
        this.elems.chatInput.placeholder = "Frag dein PDF";
        this.elems.sendButton.disabled = false;
        if (!this.currentChatId) this.createNewChat();
      } else {
        this.secondPDF = file.name;
      }

      this.updateSourceOptions();
      this.showPDFPreview(num, file.name);
    } catch {
      console.error("Upload fehlgeschlagen");
    }
  }

  // Startet neuen Chat (leert alles)
  createNewChat() {
    this.currentPDF = null;
    this.secondPDF = null;

    // Upload-Bereiche zurücksetzen
    this.elems.uploadArea1.style.display = "block";
    this.elems.uploadArea2.style.display = "block";
    this.elems.uploadedFile1.style.display = "none";
    this.elems.uploadedFile2.style.display = "none";
    this.elems.fileInput1.value = "";
    this.elems.fileInput2.value = "";

    // PDF-Vorschau zurücksetzen
    const object1 = document.getElementById("pdfObject1");
    if (object1) object1.data = "";

    const object2 = document.getElementById("pdfObject2");
    if (object2) object2.data = "";

    // Eingabe sperren bis wieder was hochgeladen wird
    this.elems.chatInput.disabled = true;
    this.elems.chatInput.placeholder = "Upload ein PDF, um zu starten...";
    this.elems.sendButton.disabled = true;

    // Neuen Chat anlegen
    const newId = Date.now().toString();
    this.chats.push({
      id: newId,
      title: `Chat ${this.chats.length + 1}`,
      messages: [],
      updatedAt: new Date().toISOString(),
    });
    this.currentChatId = newId;
    this.elems.chatMessages.innerHTML = "";
    this.renderChatList();
  }

  // Radio-Buttons für Quelle aktivieren/deaktivieren
  updateSourceOptions() {
    const [o1, o2, oBoth] = this.elems.sourceRadios;
    if (!this.secondPDF) {
      o1.disabled = false;
      o1.checked = true;
      o2.disabled = true;
      oBoth.disabled = true;
      this.source = "1";
    } else {
      o1.disabled = false;
      o2.disabled = false;
      oBoth.disabled = false;
    }
  }

  // Zeigt das PDF im Vorschaufenster an
  showPDFPreview(num, fileName) {
    const objectEl = document.getElementById(`pdfObject${num}`);
    if (!objectEl) return;

    objectEl.data = fileName ? `/pdf/${fileName}` : "";
  }

  // Nachricht senden
  async sendMessage() {
    const message = this.elems.chatInput.value.trim();
    if (!message || !this.currentPDF) return;

    this.addMessage("user", message);
    this.elems.chatInput.value = "";

    this.showTypingIndicator();

    try {
      const res = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: message, source: this.source }),
      });
      if (!res.ok) throw new Error();

      const data = await res.json();
      this.hideTypingIndicator();

      // Letzte Antwort vom Bot suchen
      const chatHistory = data.chat || [];
      const lastAssistantMsg = [...chatHistory].reverse().find((m) => m.role === "Assistant");
      this.addMessage("assistant", lastAssistantMsg?.text || "Keine Antwort erhalten.");

      // Speichern im Verlauf
      this.saveMessageToChat("User", message);
      this.saveMessageToChat("Assistant", lastAssistantMsg?.text || "");
    } catch (e) {
      this.hideTypingIndicator();
      this.addMessage("assistant", "Fehler bei der Anfrage. Bitte erneut versuchen.");
      console.error("API Fehler:", e);
    }
  }

  // Fügt neue Nachricht im Chatfenster hinzu
  addMessage(sender, text) {
    this.elems.chatMessages.insertAdjacentHTML(
      "beforeend",
      `
      <div class="message ${sender.toLowerCase()}">
        <div class="message-avatar">${sender === "user" ? "U" : "AI"}</div>
        <div class="message-content">${text}</div>
      </div>`
    );
    this.elems.chatMessages.scrollTop = this.elems.chatMessages.scrollHeight;
  }

  // Nachricht ins Speicherobjekt schreiben
  saveMessageToChat(role, text) {
    if (!this.currentChatId) return;
    const chat = this.chats.find((c) => c.id === this.currentChatId);
    if (!chat) return;
    chat.messages.push({ sender: role, text, timestamp: new Date().toISOString() });
    chat.updatedAt = new Date().toISOString();
  }

  // Lädt alten Chat wieder rein
  loadChat(chatId) {
    const chat = this.chats.find((c) => c.id === chatId);
    if (!chat) return;
    this.currentChatId = chatId;
    this.elems.chatMessages.innerHTML = "";
    chat.messages.forEach((m) => this.addMessage(m.sender, m.text));
    this.renderChatList();
  }

  // Zeigt alle alten Chats in der Liste
  renderChatList() {
    this.elems.chatList.innerHTML = "";
    this.chats.forEach((chat) => {
      const activeClass = chat.id === this.currentChatId ? "active" : "";
      this.elems.chatList.insertAdjacentHTML(
        "beforeend",
        `
        <div class="chat-item ${activeClass}" data-id="${chat.id}">
          <div class="chat-title">${chat.title}</div>
          <div class="chat-date">${new Date(chat.updatedAt).toLocaleString()}</div>
        </div>`
      );
    });
    Array.from(this.elems.chatList.children).forEach((div) =>
      div.addEventListener("click", () => this.loadChat(div.dataset.id))
    );
  }

  // "Schreibt gerade..." Animation anzeigen
  showTypingIndicator() {
  this.elems.chatMessages.insertAdjacentHTML(
    "beforeend",
    `
    <div class="message assistant" id="typing-indicator">
      <div class="message-avatar">AI</div>
      <div class="message-content">...</div>
    </div>`
  );
  this.elems.chatMessages.scrollTop = this.elems.chatMessages.scrollHeight;
}

  // "Schreibt gerade..." entfernen
  hideTypingIndicator() {
    document.getElementById("typing-indicator")?.remove();
  }

  // Exportiert den aktuellen Chat als JSON
  exportCurrentChatAsJSON() {
    if (!this.currentChatId) return;
    const chat = this.chats.find((c) => c.id === this.currentChatId);
    if (!chat) return;
    const jsonStr = JSON.stringify(chat, null, 2);
    this.downloadJSON(jsonStr, `${chat.title || "chat"}.json`);
  }

  // Lädt eine Datei runter
  downloadJSON(jsonStr, filename) {
    const blob = new Blob([jsonStr], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  }

  // Exportiert die Keypoints für ein PDF
  exportKeypoints() {
    if (!this.currentPDF) return;
    const url = "/download_keypoints";
    const a = document.createElement("a");
    a.href = url;
    a.download = `${this.currentPDF}_keypoints.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }
}

// Startet alles, wenn die Seite geladen ist
document.addEventListener("DOMContentLoaded", () => new PDFChatApp());
