import streamlit as st
import os
from dotenv import load_dotenv
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import time
import shutil

# LangChain components
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever

# --- CẤU HÌNH & THIẾT LẬP BAN ĐẦU ---
load_dotenv()
# (Tùy chỉnh đường dẫn Tesseract nếu cần, ví dụ cho Windows)
pytesseract.pytesseract.tesseract_cmd = r'E:\university\V.1\chuyendeCNTT\Tesseract\tesseract.exe'

VECTOR_STORES_MAIN_DIR = "vector_stores"
PDF_UPLOADS_DIR = "pdf_uploads"

os.makedirs(VECTOR_STORES_MAIN_DIR, exist_ok=True)
os.makedirs(PDF_UPLOADS_DIR, exist_ok=True)

# --- CSS TÙY CHỈNH CHO GIAO DIỆN MINIMALIST ---
st.set_page_config(page_title="Trợ lý AI Phân tích", layout="wide")


def load_css():
    st.markdown("""
        <style>
            /* Màu nền chính */
            .stApp {
                background-color: #1E1E1E;
            }

            /* Bo góc và đổ bóng cho các container */
            [data-testid="stSidebar"], [data-testid="stExpander"], .stChatMessage, [data-testid="stButton"] > button {
                border-radius: 10px;
            }

            [data-testid="stSidebar"] {
                 background-color: #252526;
                 border-right: 1px solid #444;
            }

            /* Nút bấm */
            [data-testid="stButton"] > button {
                border: 1px solid #444;
            }

            /* Khu vực tải file */
            [data-testid="stFileUploader"] {
                border: 2px dashed #444;
                background-color: #252526;
                padding: 25px;
            }
            [data-testid="stFileUploader"] > label {
                font-weight: bold;
                color: #FAFAFA;
            }
        </style>
    """, unsafe_allow_html=True)


load_css()


# --- CÁC HÀM BACKEND ---

@st.cache_resource
def load_models():
    """Tải và cache các mô hình AI."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    llm = GoogleGenerativeAI(model="gemini-2.0-flash-lite-001", google_api_key=os.getenv("GOOGLE_API_KEY"))
    return embeddings, llm


def extract_text_from_pdf_with_ocr(pdf_path):
    """Trích xuất văn bản từ file PDF, tự động áp dụng OCR cho các trang scan."""
    with st.spinner(f"Đang trích xuất văn bản từ '{os.path.basename(pdf_path)}'..."):
        doc = fitz.open(pdf_path)
        full_text = ""
        for page_num, page in enumerate(doc):
            digital_text = page.get_text()
            if len(digital_text.strip()) < 100:
                st.sidebar.info(f"Trang {page_num + 1} đang được OCR...")
                try:
                    pix = page.get_pixmap(dpi=300)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    ocr_text = pytesseract.image_to_string(img, lang='vie+eng')
                    full_text += ocr_text + "\n"
                except Exception as e:
                    st.sidebar.error(f"Lỗi OCR trang {page_num + 1}: {e}")
            else:
                full_text += digital_text + "\n"
    return full_text


def create_knowledge_base(pdf_path, embeddings, vector_store_path):
    """Học file PDF và lưu vào cơ sở dữ liệu vector của luồng chat hiện tại."""
    document_text = extract_text_from_pdf_with_ocr(pdf_path)
    if not document_text.strip():
        st.error("Không trích xuất được văn bản nào từ file PDF.")
        return

    with st.spinner("Đang xây dựng cơ sở tri thức..."):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        chunks = text_splitter.create_documents([document_text], metadatas=[{"source": pdf_path}])
        db = Chroma.from_documents(chunks, embeddings, persist_directory=vector_store_path)
        db.persist()
    st.success(f"Đã học thành công file: {os.path.basename(pdf_path)}")


def create_advanced_qa_chain(embeddings, llm, vector_store_path):
    """Tạo chuỗi Q&A từ cơ sở dữ liệu vector của luồng chat hiện tại."""
    if not os.path.exists(vector_store_path) or not any(os.scandir(vector_store_path)):
        return None
    vectorstore = Chroma(persist_directory=vector_store_path, embedding_function=embeddings)
    retriever = MultiQueryRetriever.from_llm(retriever=vectorstore.as_retriever(search_kwargs={"k": 5}), llm=llm)
    prompt_template = """Bạn là một trợ lý AI chuyên gia, có khả năng phân tích và suy luận sâu sắc.
    Nhiệm vụ của bạn là trả lời câu hỏi của người dùng bằng cách TỔNG HỢP, SUY LUẬN và RÚT RA CÁC KẾT LUẬN LOGIC từ những thông tin được cung cấp trong phần 'NGỮ CẢNH'.
    HƯỚNG DẪN SUY LUẬN:
    - Đừng chỉ trích xuất thông tin. Hãy hành động như một chuyên gia, kết nối các ý tưởng, chỉ ra các hàm ý và cung cấp một câu trả lời toàn diện, ngay cả khi câu trả lời không được viết nguyên văn trong tài liệu.
    - Đưa ra các phân tích "dựa trên" và "suy ra từ" các dữ kiện trong ngữ cảnh.
    - Tuy nhiên, mọi suy luận của bạn phải được bám rễ vững chắc vào các dữ kiện có trong ngữ cảnh. Không được bịa đặt thông tin.
    - Nếu ngữ cảnh hoàn toàn không liên quan hoặc không đủ để trả lời, hãy nói rõ điều đó.
    - Luôn trả lời bằng tiếng Việt một cách chuyên nghiệp và sâu sắc.
    - Về bố cục của câu trả lời, nếu nó dài và có nhiều ý, hãy trình bày tách đoạn và xuống dòng một cách khoa học.
    NGỮ CẢNH: {context}
    CÂU HỎI: {question}
    CÂU TRẢ LỜI CỦA CHUYÊN GIA (đã phân tích và suy luận):"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever,
                                           return_source_documents=True, chain_type_kwargs={"prompt": PROMPT})
    return qa_chain


# --- TIÊU ĐỀ & KHỞI TẠO ---
st.title("🚀 Trợ lý AI Phân tích Tài liệu")

if 'chats' not in st.session_state:
    st.session_state.chats = {}
if 'active_chat_id' not in st.session_state:
    st.session_state.active_chat_id = None
if 'confirm_delete' not in st.session_state:
    st.session_state.confirm_delete = None

if not st.session_state.chats:
    first_chat_id = f"Dự án {int(time.time())}"
    st.session_state.chats[first_chat_id] = {"title": "Dự án mới", "messages": [],
                                             "vector_store_path": os.path.join(VECTOR_STORES_MAIN_DIR, first_chat_id),
                                             "learned_files": set(), "latest_sources": "Chưa có câu trả lời nào."}
    st.session_state.active_chat_id = first_chat_id

embeddings, llm = load_models()

# --- SIDEBAR: ĐIỀU KHIỂN & QUẢN LÝ DỰ ÁN ---
with st.sidebar:
    st.header("⚙️ Bảng điều khiển")
    if st.button("➕ Dự án mới", use_container_width=True):
        new_chat_id = f"Dự án {int(time.time())}"
        st.session_state.chats[new_chat_id] = {"title": "Dự án mới không tên", "messages": [],
                                               "vector_store_path": os.path.join(VECTOR_STORES_MAIN_DIR, new_chat_id),
                                               "learned_files": set(), "latest_sources": "Chưa có câu trả lời nào."}
        st.session_state.active_chat_id = new_chat_id
        st.rerun()

    st.subheader("📁 Các dự án")
    chat_ids = sorted(st.session_state.chats.keys(), reverse=True)
    for chat_id in chat_ids:
        col1, col2 = st.columns([4, 1])
        chat_title = st.session_state.chats[chat_id]['title']
        button_type = "primary" if chat_id == st.session_state.active_chat_id else "secondary"
        with col1:
            if st.button(chat_title, key=f"switch_{chat_id}", use_container_width=True, type=button_type):
                st.session_state.active_chat_id = chat_id
                st.session_state.confirm_delete = None  # Hủy xóa khi chuyển luồng
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"delete_{chat_id}", use_container_width=True):
                st.session_state.confirm_delete = chat_id
                st.rerun()

    if st.session_state.confirm_delete:
        chat_to_delete_id = st.session_state.confirm_delete
        chat_to_delete_title = st.session_state.chats[chat_to_delete_id]['title']
        st.warning(f"Bạn có chắc muốn xóa vĩnh viễn dự án '{chat_to_delete_title}' không?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Xác nhận", use_container_width=True, type="primary"):
                vector_path_to_delete = st.session_state.chats[chat_to_delete_id]['vector_store_path']
                del st.session_state.chats[chat_to_delete_id]
                if os.path.exists(vector_path_to_delete):
                    shutil.rmtree(vector_path_to_delete)

                st.session_state.confirm_delete = None
                if st.session_state.active_chat_id == chat_to_delete_id:
                    st.session_state.active_chat_id = next(iter(st.session_state.chats), None)
                st.rerun()
        with col2:
            if st.button("❌ Hủy", use_container_width=True):
                st.session_state.confirm_delete = None
                st.rerun()

    st.divider()

    if st.session_state.active_chat_id:
        active_chat = st.session_state.chats[st.session_state.active_chat_id]
        st.subheader(f"Quản lý tài liệu")
        uploaded_file = st.file_uploader("Kéo & thả file PDF vào đây", type="pdf", label_visibility="collapsed")
        if uploaded_file:
            file_path = os.path.join(PDF_UPLOADS_DIR, uploaded_file.name)
            if not os.path.exists(file_path):
                with open(file_path, "wb") as f: f.write(uploaded_file.getbuffer())
            if st.button("Học File Này", use_container_width=True):
                create_knowledge_base(file_path, embeddings, active_chat["vector_store_path"])
                active_chat["learned_files"].add(uploaded_file.name)

        st.write("Các file đã học:")
        if active_chat["learned_files"]:
            for file_name in active_chat["learned_files"]: st.markdown(f"- `{file_name}`")
        else:
            st.write("Chưa có file nào.")

    st.divider()

    if st.session_state.active_chat_id:
        st.subheader("📚 Nguồn trích dẫn")
        st.container(height=300).markdown(st.session_state.chats[st.session_state.active_chat_id]["latest_sources"],
                                          unsafe_allow_html=True)

# --- KHU VỰC CHÍNH: GIAO DIỆN CHAT ---
active_chat_id = st.session_state.active_chat_id
if active_chat_id:
    active_chat = st.session_state.chats[active_chat_id]
    new_title = st.text_input("Tên dự án:", value=active_chat["title"], label_visibility="collapsed")
    if new_title != active_chat["title"]:
        st.session_state.chats[active_chat_id]["title"] = new_title
        st.rerun()

    # Hiển thị lịch sử chat
    for message in active_chat["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    qa_chain = create_advanced_qa_chain(embeddings, llm, active_chat["vector_store_path"])

    if user_question := st.chat_input("Đặt câu hỏi về các tài liệu trong dự án này..."):
        if not qa_chain:
            st.warning("Dự án này chưa có kiến thức. Vui lòng 'Học' một file PDF ở thanh bên.")
            st.stop()

        # Lấy tiêu đề cho chat nếu đây là tin nhắn đầu tiên
        if not active_chat["messages"]:
            st.session_state.chats[active_chat_id]["title"] = user_question

        active_chat["messages"].append({"role": "user", "content": user_question})

        with st.spinner("AI đang phân tích và suy luận..."):
            response = qa_chain.invoke(user_question)
            answer = response['result']

            sources_text = ""
            if 'source_documents' in response and response['source_documents']:
                for doc in response['source_documents']:
                    source_name = os.path.basename(doc.metadata.get('source', 'Không rõ'))
                    content = doc.page_content.replace('\n', ' ').strip()
                    sources_text += f"<div style='border: 1px solid #444; border-radius: 5px; padding: 10px; margin-bottom: 10px;'><b>Nguồn:</b> <code>{source_name}</code><br><small><b>Nội dung:</b> ...{content[:250]}...</small></div>"
            else:
                sources_text = "Không tìm thấy nguồn trích dẫn nào."

            active_chat["latest_sources"] = sources_text
            active_chat["messages"].append({"role": "assistant", "content": answer})
            st.rerun()

else:
    st.info("Chào mừng bạn! Hãy tạo một 'Dự án mới' ở thanh bên để bắt đầu.")