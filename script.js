
class PDFChatApp {
  constructor() {
    this.currentChatId = null;
    this.chats = [];
    this.currentPDF = null;
    this.secondPDF = null;
    this.source = "1";
    this.elems = {
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
    this.initUploadSections();
    this.bindEvents();
    this.createNewChat();
  }

  initUploadSections() {
    const container = document.getElementById("uploadContainer");
    const template = document.getElementById("uploadTemplate").innerHTML;
    [1, 2].forEach((num) => {
      const html = template.replaceAll("{NUM}", num);
      const wrapper = document.createElement("div");
      wrapper.innerHTML = html;
      container.appendChild(wrapper.firstElementChild);
      document.querySelector(`#uploadArea${num} .upload-text`).textContent =
        num === 1 ? "PDF hier ablegen oder klicken" : "Zweites PDF hier ablegen oder klicken";
    });

    this.elems.uploadArea1 = document.getElementById("uploadArea1");
    this.elems.fileInput1 = document.getElementById("fileInput1");
    this.elems.uploadArea2 = document.getElementById("uploadArea2");
    this.elems.fileInput2 = document.getElementById("fileInput2");
  }

  bindEvents() {
    [1, 2].forEach((num) => this.setupUpload(num));
    this.elems.sendButton.addEventListener("click", () => this.sendMessage());
    this.elems.chatInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        this.sendMessage();
      }
    });
    this.elems.newChatBtn.addEventListener("click", () => this.createNewChat());
    this.elems.sourceRadios.forEach((r) =>
      r.addEventListener("change", (e) => (this.source = e.target.value))
    );
    this.elems.exportChatBtn?.addEventListener("click", () => this.exportCurrentChatAsJSON());
    this.elems.exportKeypointsBtn?.addEventListener("click", () => this.exportKeypoints());
  }

  setupUpload(num) {
    const uploadArea = this.elems[`uploadArea${num}`];
    const fileInput = this.elems[`fileInput${num}`];

    uploadArea.onclick = () => fileInput.click();
    fileInput.onchange = (e) => {
      const file = e.target.files[0];
      if (file?.type === "application/pdf") this.handleFile(num, file);
    };
  }

  async handleFile(num, file) {
    const formData = new FormData();
    formData.append("file", file);
    try {
      const url = num === 1 ? "/upload" : "/upload2";
      const res = await fetch(url, { method: "POST", body: formData });
      if (!res.ok) throw new Error();

      this.elems[`uploadArea${num}`].style.display = "none";

      if (num === 1) this.currentPDF = file.name;
      else this.secondPDF = file.name;

      this.updateSourceOptions();
      this.showPDFPreview(num, file.name);
    } catch {
      console.error("Upload fehlgeschlagen");
    }
  }

  createNewChat() {
    this.currentPDF = null;
    this.secondPDF = null;
    this.elems.uploadArea1.style.display = "block";
    this.elems.uploadArea2.style.display = "block";
    this.elems.fileInput1.value = "";
    this.elems.fileInput2.value = "";
    document.getElementById("pdfObject1").data = "";
    document.getElementById("pdfObject2").data = "";
    document.getElementById("pdfPreview1").classList.remove("visible");
    document.getElementById("pdfPreview2").classList.remove("visible");
    this.elems.chatInput.value = "";
    this.elems.sendButton.disabled = false;

    const newId = Date.now().toString();
    this.chats.push({ id: newId, title: `Chat ${this.chats.length + 1}`, messages: [], updatedAt: new Date().toISOString() });
    this.currentChatId = newId;
    this.elems.chatMessages.innerHTML = "";
    this.renderChatList();
  }

  updateSourceOptions() {
    const [o1, o2, oBoth] = this.elems.sourceRadios;
    if (!this.secondPDF) {
      o1.disabled = false;
      o1.checked = true;
      o2.disabled = true;
      oBoth.disabled = true;
      this.source = "1";
    } else {
      o1.disabled = o2.disabled = oBoth.disabled = false;
      const radio = document.getElementById(`source${this.source.charAt(0)}`);
      if (radio) radio.checked = true;
    }
  }

  showPDFPreview(num, fileName) {
    const previewDiv = document.getElementById(`pdfPreview${num}`);
    const objectEl = document.getElementById(`pdfObject${num}`);
    if (!objectEl || !previewDiv) return;
    objectEl.data = fileName ? `/pdf/${fileName}` : "";
    previewDiv.classList.toggle("visible", Boolean(fileName));
  }

  renderChatList() {
    this.elems.chatList.innerHTML = "";
    this.chats.forEach((chat) => {
      const activeClass = chat.id === this.currentChatId ? "active" : "";
      this.elems.chatList.insertAdjacentHTML(
        "beforeend",
        `<div class="chat-item ${activeClass}" data-id="${chat.id}">
          <div class="chat-title">${chat.title}</div>
          <div class="chat-date">${new Date(chat.updatedAt).toLocaleString()}</div>
        </div>`
      );
    });
    Array.from(this.elems.chatList.children).forEach((div) =>
      div.addEventListener("click", () => this.loadChat(div.dataset.id))
    );
  }

  loadChat(chatId) {
    const chat = this.chats.find((c) => c.id === chatId);
    if (!chat) return;
    this.currentChatId = chatId;
    this.elems.chatMessages.innerHTML = "";
    chat.messages.forEach((m) => this.addMessage(m.sender, m.text));
    this.renderChatList();
  }

  addMessage(sender, text) {
    this.elems.chatMessages.insertAdjacentHTML(
      "beforeend",
      `<div class="message ${sender.toLowerCase()}">
        <div class="message-avatar">${sender[0].toUpperCase()}</div>
        <div class="message-content">${text}</div>
      </div>`
    );
    this.elems.chatMessages.scrollTop = this.elems.chatMessages.scrollHeight;
  }

  async sendMessage() {
    const message = this.elems.chatInput.value.trim();
    if (!message) return;
    this.addMessage("user", message);
    this.elems.chatInput.value = "";
    this.addMessage("assistant", "...");
    try {
      const res = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: message, source: this.source }),
      });
      if (!res.ok) throw new Error();
      const data = await res.json();
      this.elems.chatMessages.lastElementChild.remove();
      const lastAssistantMsg = [...data.chat].reverse().find((m) => m.role === "Assistant");
      this.addMessage("assistant", lastAssistantMsg?.text || "Keine Antwort erhalten.");
      this.saveMessageToChat("User", message);
      this.saveMessageToChat("Assistant", lastAssistantMsg?.text || "");
    } catch (e) {
      this.elems.chatMessages.lastElementChild.remove();
      this.addMessage("assistant", "Fehler bei der Anfrage. Bitte erneut versuchen.");
    }
  }

  saveMessageToChat(role, text) {
    const chat = this.chats.find((c) => c.id === this.currentChatId);
    if (!chat) return;
    chat.messages.push({ sender: role, text, timestamp: new Date().toISOString() });
    chat.updatedAt = new Date().toISOString();
  }

  exportCurrentChatAsJSON() {
    const chat = this.chats.find((c) => c.id === this.currentChatId);
    if (!chat) return;
    const blob = new Blob([JSON.stringify(chat, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${chat.title}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  exportKeypoints() {
    if (!this.currentPDF) return;
    const a = document.createElement("a");
    a.href = "/download_keypoints";
    a.download = `${this.currentPDF}_keypoints.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }
}

document.addEventListener("DOMContentLoaded", () => new PDFChatApp());
