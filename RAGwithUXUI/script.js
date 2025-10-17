document.addEventListener('DOMContentLoaded', () => {
    // --- KHAI BÁO BIẾN & LẤY CÁC THÀNH PHẦN GIAO DIỆN ---
    const API_URL = 'http://127.0.0.1:5001';

    // Sidebar
    const sidebar = document.getElementById('sidebar');
    const toggleSidebarBtn = document.getElementById('toggle-sidebar-btn');
    const newChatBtn = document.getElementById('new-chat-btn');
    const projectList = document.getElementById('project-list');
    const pdfDropZone = document.getElementById('pdf-drop-zone');
    const pdfUploadInput = document.getElementById('pdf-upload');
    const pdfUploadLabelSpan = document.querySelector('#pdf-upload-label span');
    const learnPdfBtn = document.getElementById('learn-pdf-btn');
    const learnedFilesList = document.getElementById('learned-files-list');
    const citationContainer = document.getElementById('citation-container');

    // Main content
    const mainContent = document.getElementById('main-content');
    const projectTitleHeader = document.getElementById('project-title-header');
    const chatHistory = document.getElementById('chat-history');
    const imageUploadLabel = document.getElementById('image-upload-label');
    const imageUploadInput = document.getElementById('image-upload');
    const chatInput = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-btn');
    const imagePreviewContainer = document.getElementById('image-preview-container');

    // Quản lý trạng thái
    let state = {
        chats: {},
        activeChatId: null,
        stagedImage: null,
        renamingChatId: null
    };

    // --- CÁC HÀM XỬ LÝ ---

    // Hàm gọi API backend đã được nâng cấp để xử lý lỗi
    async function apiCall(endpoint, method = 'POST', body = null) {
        try {
            const options = { method, headers: {} };
            if (body) {
                if (body instanceof FormData) {
                    options.body = body;
                } else {
                    options.headers['Content-Type'] = 'application/json';
                    options.body = JSON.stringify(body);
                }
            }
            const response = await fetch(`${API_URL}${endpoint}`, options);
            const responseData = await response.json();
            if (!response.ok) {
                return { success: false, error: responseData.error || `Lỗi máy chủ: ${response.status}` };
            }
            return { success: true, data: responseData };
        } catch (error) {
            console.error('API Call Error:', error);
            return { success: false, error: error.message };
        }
    }

    // Cập nhật trạng thái nút Học File
    function setButtonState(button, state, defaultText = 'Học File Này') {
        const textSpan = button.querySelector('.btn-text');
        const iconSpan = button.querySelector('.btn-icon');
        button.disabled = false;
        iconSpan.innerHTML = '';
        textSpan.textContent = defaultText;

        if (state === 'loading') {
            button.disabled = true;
            textSpan.textContent = 'Đang học...';
            iconSpan.innerHTML = '<div class="spinner"></div>';
        } else if (state === 'success') {
            button.disabled = true;
            textSpan.textContent = 'Thành công!';
            iconSpan.innerHTML = '<span class="success">✓</span>';
        } else if (state === 'error') {
            button.disabled = true;
            textSpan.textContent = 'Thất bại!';
            iconSpan.innerHTML = '<span class="error">✗</span>';
        }
    }

    // Hiển thị tin nhắn trong chat (sử dụng marked.js)
    function addMessageToHistory(role, text, imageUrl = null) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('chat-message', `${role}-message`);

        if (text) {
            messageDiv.innerHTML = marked.parse(text);
        }
        if (imageUrl) {
            const imgNode = document.createElement('img');
            imgNode.src = imageUrl;
            imgNode.style.maxWidth = '300px';
            imgNode.style.borderRadius = '10px';
            imgNode.style.marginTop = text ? '10px' : '0';

            if (!text) {
                messageDiv.appendChild(imgNode);
            } else {
                const firstChild = messageDiv.firstChild;
                if (firstChild) {
                    messageDiv.insertBefore(imgNode, firstChild.nextSibling);
                } else {
                    messageDiv.appendChild(imgNode);
                }
            }
        }

        chatHistory.appendChild(messageDiv);
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }

    // Cập nhật giao diện dựa trên luồng chat đang hoạt động
    function renderActiveChat() {
        chatHistory.innerHTML = '';
        citationContainer.innerHTML = '<small>Nguồn trích dẫn sẽ hiện ở đây.</small>';
        learnedFilesList.innerHTML = '<small>Chưa có file nào trong dự án này.</small>';
        if (!state.activeChatId || !state.chats[state.activeChatId]) return;

        const activeChat = state.chats[state.activeChatId];
        projectTitleHeader.textContent = activeChat.title;
        activeChat.messages.forEach(msg => addMessageToHistory(msg.role, msg.text, msg.image));

        if (activeChat.learned_files && activeChat.learned_files.size > 0) {
            learnedFilesList.innerHTML = '';
            activeChat.learned_files.forEach(file => {
                const fileItem = document.createElement('div');
                fileItem.textContent = `- ${file}`;
                learnedFilesList.appendChild(fileItem);
            });
        }
    }

    // Cập nhật danh sách dự án
    function renderProjectList() {
        projectList.innerHTML = '';
        Object.keys(state.chats).forEach(chatId => {
            const item = document.createElement('div');
            item.className = 'project-item';
            item.dataset.chatId = chatId;
            if (chatId === state.activeChatId) item.classList.add('active');

            if (state.renamingChatId === chatId) {
                const input = document.createElement('input');
                input.type = 'text';
                input.className = 'rename-input';
                input.value = state.chats[chatId].title;
                item.appendChild(input);
                setTimeout(() => { input.focus(); input.select(); }, 0);
                const saveRename = () => {
                    const newTitle = input.value.trim();
                    if (newTitle) state.chats[chatId].title = newTitle;
                    state.renamingChatId = null;
                    renderProjectList();
                    renderActiveChat();
                };
                input.onblur = saveRename;
                input.onkeydown = (e) => {
                    if (e.key === 'Enter') input.blur();
                    if (e.key === 'Escape') { state.renamingChatId = null; renderProjectList(); }
                };
            } else {
                const titleSpan = document.createElement('span');
                titleSpan.className = 'project-item-title';
                titleSpan.textContent = state.chats[chatId].title;
                item.appendChild(titleSpan);
                const menuBtn = document.createElement('button');
                menuBtn.className = 'icon-btn';
                menuBtn.innerHTML = '⋮';
                menuBtn.onclick = (e) => {
                    e.stopPropagation();
                    document.querySelectorAll('.project-menu').forEach(menu => menu.remove());
                    createProjectMenu(chatId, item);
                };
                item.appendChild(menuBtn);
            }

            item.onclick = (e) => {
                if (e.target.tagName === 'INPUT' || state.renamingChatId) return;
                state.activeChatId = chatId;
                renderProjectList();
                renderActiveChat();
            };
            projectList.appendChild(item);
        });
    }

    // Tạo menu tác vụ cho dự án
    function createProjectMenu(chatId, parentItem) {
        const menu = document.createElement('div');
        menu.className = 'project-menu';
        const renameBtn = document.createElement('button');
        renameBtn.className = 'menu-item';
        renameBtn.textContent = '✏️ Đổi tên';
        renameBtn.onclick = () => {
            menu.remove();
            state.renamingChatId = chatId;
            renderProjectList();
        };
        const deleteBtn = document.createElement('button');
        deleteBtn.className = 'menu-item';
        deleteBtn.textContent = '🗑️ Xóa';
        deleteBtn.onclick = () => {
            menu.remove();
            if (confirm(`Bạn có chắc muốn xóa dự án "${state.chats[chatId].title}"?`)) {
                delete state.chats[chatId];
                if (state.activeChatId === chatId) {
                    state.activeChatId = Object.keys(state.chats)[0] || null;
                }
                renderProjectList();
                renderActiveChat();
            }
        };
        menu.appendChild(renameBtn);
        menu.appendChild(deleteBtn);
        parentItem.appendChild(menu);
    }

    // Xử lý logic gửi tin nhắn (đã có xử lý lỗi)
    async function handleSendMessage() {
        const question = chatInput.value.trim();
        const image = state.stagedImage;
        if (!question && !image) return;

        addMessageToHistory('user', question, image);
        state.chats[state.activeChatId].messages.push({ role: 'user', text: question, image });

        if(state.chats[state.activeChatId].messages.length === 1 && question) {
            state.chats[state.activeChatId].title = question.substring(0, 30);
            renderProjectList();
            projectTitleHeader.textContent = state.chats[state.activeChatId].title;
        }

        const body = { chatId: state.activeChatId, question: question, image: image };
        addMessageToHistory('assistant', 'Đang suy nghĩ...');

        const responseWrapper = await apiCall('/ask', 'POST', body);

        chatHistory.removeChild(chatHistory.lastChild);

        if (responseWrapper.success) {
            const response = responseWrapper.data;
            addMessageToHistory('assistant', response.answer);
            state.chats[state.activeChatId].messages.push({ role: 'assistant', text: response.answer });
            citationContainer.innerHTML = '';
            if (response.sources && response.sources.length > 0) {
                response.sources.forEach(src => {
                    const item = document.createElement('div');
                    item.className = 'citation-item';
                    item.innerHTML = `<b>Nguồn:</b> <code>${src.source}</code><br><small><b>Nội dung:</b> ...${src.content.substring(0, 150)}...</small>`;
                    citationContainer.appendChild(item);
                });
            } else {
                citationContainer.innerHTML = '<small>Không tìm thấy nguồn trích dẫn liên quan.</small>';
            }
        } else {
            addMessageToHistory('assistant', `**Rất tiếc, đã có lỗi xảy ra:**\n\n${responseWrapper.error}`);
        }

        chatInput.value = '';
        state.stagedImage = null;
        imagePreviewContainer.innerHTML = '';
    }

    // Khởi tạo ứng dụng
    function initialize() {
        const firstChatId = `Dự án ${Date.now()}`;
        state.chats[firstChatId] = { title: 'Dự án mới', messages: [], learned_files: new Set() };
        state.activeChatId = firstChatId;
        renderProjectList();
        renderActiveChat();
    }

    // --- GÁN CÁC SỰ KIỆN ---

    toggleSidebarBtn.onclick = () => {
        sidebar.classList.toggle('collapsed');
        toggleSidebarBtn.classList.toggle('collapsed');
    };

    newChatBtn.onclick = () => {
        const newChatId = `Dự án ${Date.now()}`;
        state.chats[newChatId] = { title: 'Dự án mới', messages: [], learned_files: new Set() };
        state.activeChatId = newChatId;
        renderProjectList();
        renderActiveChat();
    };

    sendBtn.onclick = handleSendMessage;
    chatInput.onkeydown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSendMessage();
        }
    };

    document.addEventListener('paste', (event) => {
        const items = (event.clipboardData || event.originalEvent.clipboardData).items;
        for (const item of items) {
            if (item.type.indexOf('image') === 0) {
                const blob = item.getAsFile();
                const reader = new FileReader();
                reader.onload = (e) => {
                    state.stagedImage = e.target.result;
                    imagePreviewContainer.innerHTML = `<img src="${state.stagedImage}" alt="Ảnh chờ gửi"><button id="remove-img-btn" class="icon-btn">&times;</button>`;
                    document.getElementById('remove-img-btn').onclick = () => {
                        state.stagedImage = null;
                        imagePreviewContainer.innerHTML = '';
                    };
                };
                reader.readAsDataURL(blob);
                event.preventDefault();
                return;
            }
        }
    });

    imageUploadLabel.onclick = () => imageUploadInput.click();
    imageUploadInput.onchange = (event) => {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                state.stagedImage = e.target.result;
                imagePreviewContainer.innerHTML = `<img src="${state.stagedImage}" alt="Ảnh chờ gửi"><button id="remove-img-btn" class="icon-btn">&times;</button>`;
                document.getElementById('remove-img-btn').onclick = () => {
                    state.stagedImage = null;
                    imagePreviewContainer.innerHTML = '';
                };
            };
            reader.readAsDataURL(file);
        }
    };

    pdfDropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        pdfDropZone.classList.add('drag-over');
    });
    pdfDropZone.addEventListener('dragleave', () => pdfDropZone.classList.remove('drag-over'));
    pdfDropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        pdfDropZone.classList.remove('drag-over');
        const file = e.dataTransfer.files[0];
        if (file && file.type === 'application/pdf') {
            pdfUploadInput.files = e.dataTransfer.files;
            pdfUploadInput.dispatchEvent(new Event('change'));
        }
    });

    pdfUploadInput.onchange = (event) => {
        const file = event.target.files[0];
        if (file) {
            pdfUploadLabelSpan.textContent = file.name;
            learnPdfBtn.style.display = 'block';
        }
    };

    learnPdfBtn.onclick = async () => {
        const file = pdfUploadInput.files[0];
        if (!file || !state.activeChatId) return;

        setButtonState(learnPdfBtn, 'loading');

        const formData = new FormData();
        formData.append('file', file);
        formData.append('chatId', state.activeChatId);

        const response = await apiCall('/learn', 'POST', formData);

        if(response.success) {
            setButtonState(learnPdfBtn, 'success');
            state.chats[state.activeChatId].learned_files.add(file.name);
            renderActiveChat();
        } else {
            setButtonState(learnPdfBtn, 'error');
        }

        setTimeout(() => {
            setButtonState(learnPdfBtn, 'idle', 'Học File Này');
            learnPdfBtn.style.display = 'none';
            pdfUploadLabelSpan.textContent = 'Kéo & thả file PDF vào đây';
        }, 2000);
    };

    document.addEventListener('click', (e) => {
        if (!e.target.closest('.project-item')) {
            document.querySelectorAll('.project-menu').forEach(menu => menu.remove());
        }
    });

    initialize();
});