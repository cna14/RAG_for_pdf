from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import base64
import io

# LangChain components
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.messages import HumanMessage

# --- CẤU HÌNH & KHỞI TẠO ---
print("--- Khởi tạo Backend Server ---")
load_dotenv()

# Cấu hình đường dẫn Tesseract (QUAN TRỌNG: Phải chính xác)
try:
    pytesseract.pytesseract.tesseract_cmd = r'E:\university\V.1\chuyendeCNTT\Tesseract\tesseract.exe'
    print("Đã tìm thấy Tesseract OCR.")
except Exception:
    print("CẢNH BÁO: Không tìm thấy Tesseract tại đường dẫn đã chỉ định. Chức năng OCR có thể không hoạt động.")

# Định nghĩa các đường dẫn thư mục
VECTOR_STORES_MAIN_DIR = "vector_stores"
PDF_UPLOADS_DIR = "pdf_uploads"
os.makedirs(VECTOR_STORES_MAIN_DIR, exist_ok=True)
os.makedirs(PDF_UPLOADS_DIR, exist_ok=True)

# Khởi tạo Flask App và cho phép CORS
app = Flask(__name__)
CORS(app)

# Tải các mô hình AI (chỉ một lần khi server khởi động)
print("Đang tải các mô hình AI (Embedding & LLM)...")
try:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    llm = GoogleGenerativeAI(model="gemini-2.0-flash-lite-001", google_api_key=os.getenv("GOOGLE_API_KEY"))
    print("Các mô hình AI đã sẵn sàng.")
except Exception as e:
    print(f"LỖI NGHIÊM TRỌNG: Không thể tải mô hình AI. Lỗi: {e}")
    llm = None


# --- CÁC HÀM LOGIC XỬ LÝ ---

def extract_text_from_pdf_with_ocr(pdf_path):
    """Trích xuất văn bản từ file PDF, tự động áp dụng OCR cho các trang scan."""
    print(f"Bắt đầu trích xuất văn bản từ '{os.path.basename(pdf_path)}'...")
    doc = fitz.open(pdf_path)
    full_text = ""
    for page_num, page in enumerate(doc):
        digital_text = page.get_text()
        if len(digital_text.strip()) < 100:
            print(f"  - Trang {page_num + 1} đang được OCR...")
            try:
                pix = page.get_pixmap(dpi=300)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                ocr_text = pytesseract.image_to_string(img, lang='vie+eng')
                full_text += ocr_text + "\n"
            except Exception as e:
                print(f"Lỗi OCR trang {page_num + 1}: {e}")
        else:
            full_text += digital_text + "\n"
    print("Trích xuất văn bản hoàn tất.")
    return full_text


def create_knowledge_base(pdf_path, embeddings, vector_store_path):
    """Học file PDF và lưu vào cơ sở dữ liệu vector của luồng chat hiện tại."""
    document_text = extract_text_from_pdf_with_ocr(pdf_path)
    if not document_text.strip():
        raise ValueError("Không trích xuất được văn bản nào từ file PDF.")

    print("Đang xây dựng cơ sở tri thức...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = text_splitter.create_documents([document_text], metadatas=[{"source": pdf_path}])
    db = Chroma.from_documents(chunks, embeddings, persist_directory=vector_store_path)
    db.persist()
    print("Xây dựng cơ sở tri thức thành công.")


# --- CÁC ĐIỂM CUỐI API (API ENDPOINTS) ---

@app.route('/learn', methods=['POST'])
def learn_pdf():
    """Endpoint để nhận file PDF và 'học' nó."""
    print("\n[API] Nhận được yêu cầu tại /learn")
    if 'file' not in request.files or 'chatId' not in request.form:
        return jsonify({"error": "Yêu cầu không hợp lệ, thiếu file hoặc chatId."}), 400

    file = request.files['file']
    chat_id = request.form['chatId']

    file_path = os.path.join(PDF_UPLOADS_DIR, file.filename)
    file.save(file_path)

    vector_store_path = os.path.join(VECTOR_STORES_MAIN_DIR, chat_id)
    try:
        create_knowledge_base(file_path, embeddings, vector_store_path)
        return jsonify({"message": f"Học thành công file: {file.filename}"})
    except Exception as e:
        print(f"Lỗi khi học file: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/ask', methods=['POST'])
def ask_question():
    """Endpoint để nhận câu hỏi (và ảnh tùy chọn) và trả về câu trả lời."""
    print("\n[API] Nhận được yêu cầu tại /ask")
    if llm is None:
        return jsonify({"error": "Mô hình AI chưa được tải, không thể xử lý yêu cầu."}), 503

    data = request.json
    question = data.get('question', '')
    chat_id = data.get('chatId')
    image_data_url = data.get('image')

    if not chat_id:
        return jsonify({"error": "Thiếu ID của dự án (chatId)."}), 400

    vector_store_path = os.path.join(VECTOR_STORES_MAIN_DIR, chat_id)
    retrieved_docs = []
    if os.path.exists(vector_store_path) and question:
        try:
            vectorstore = Chroma(persist_directory=vector_store_path, embedding_function=embeddings)
            retriever = MultiQueryRetriever.from_llm(retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
                                                     llm=llm)
            retrieved_docs = retriever.invoke(question)
        except Exception as e:
            print(f"Lỗi khi truy xuất tài liệu: {e}")

    context_text = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])

    try:
        # PROMPT ĐÃ ĐƯỢC NÂNG CẤP TOÀN DIỆN
        prompt_template = """Bạn là một trợ lý AI chuyên gia, có khả năng phân tích đa phương thức.
        Nhiệm vụ của bạn là trả lời 'CÂU HỎI' bằng cách TỔNG HỢP, SUY LUẬN và RÚT RA KẾT LUẬN LOGIC từ những thông tin được cung cấp.

        **NGUỒN THÔNG TIN CỦA BẠN BAO GỒM:**
        1.  `NGỮ CẢNH TỪ TÀI LIỆU`: Các đoạn văn bản đã được trích xuất từ file PDF có liên quan đến câu hỏi.
        2.  `HÌNH ẢNH ĐÍNH KÈM`: Một hình ảnh do người dùng cung cấp (nếu có).

        **QUY TRÌNH TƯ DUY BẮT BUỘC:**
        1.  **Ưu tiên Hình ảnh (nếu có):** Đầu tiên, hãy phân tích kỹ `HÌNH ẢNH ĐÍNH KÈM`. Nếu hình ảnh chứa câu trả lời trực tiếp cho `CÂU HỎI`, hãy dựa vào đó làm bằng chứng chính.
        2.  **Kết hợp & Đối chiếu:** Sau đó, đối chiếu thông tin từ hình ảnh với `NGỮ CẢNH TỪ TÀI LIỆU`. Hãy kết nối các ý tưởng, chỉ ra sự tương đồng hoặc mâu thuẫn (nếu có) giữa hai nguồn thông tin để đưa ra một câu trả lời sâu sắc và toàn diện.
        3.  **Hành động như chuyên gia:** Đừng chỉ trích xuất thông tin. Hãy đưa ra các phân tích "dựa trên", "suy ra từ" các dữ kiện bạn có.
        4.  **Bám sát sự thật:** Mọi suy luận phải được bám rễ vững chắc vào các dữ kiện được cung cấp. Không được bịa đặt. Nếu cả hai nguồn đều không đủ thông tin, hãy nói rõ điều đó.

        **HƯỚNG DẪN ĐỊNH DẠNG CÂU TRẢ LỜI:**
        - Sử dụng Markdown để trình bày câu trả lời một cách rõ ràng, trực quan.
        - Dùng `**in đậm**` cho các thuật ngữ hoặc kết luận quan trọng.
        - Sử dụng danh sách gạch đầu dòng (`-`) hoặc danh sách có số thứ tự (`1.`, `2.`) để liệt kê các ý.
        - Luôn trả lời bằng tiếng Việt một cách chuyên nghiệp.

        ---
        **NGỮ CẢNH TỪ TÀI LIỆU:**
        {context}
        ---
        **CÂU HỎI:**
        {question}
        ---
        **CÂU TRẢ LỜI CỦA CHUYÊN GIA (đã phân tích và định dạng theo Markdown):**"""

        final_question = question if question else "Hãy phân tích và mô tả chi tiết nội dung của hình ảnh này. Nếu có thể, hãy liên hệ nó với các thông tin có trong ngữ cảnh tài liệu."

        final_prompt_text = prompt_template.format(context=context_text, question=final_question)

        answer = ""  # Khởi tạo biến answer

        print("Đang gửi yêu cầu đến mô hình Gemini...")

        # --- SỬA LỖI: Phân luồng xử lý cho từng trường hợp ---
        if image_data_url:
            # **Trường hợp có ảnh (đa phương thức)**
            header, encoded = image_data_url.split(",", 1)
            image_to_process = Image.open(io.BytesIO(base64.b64decode(encoded)))

            # Đóng gói vào HumanMessage
            message = HumanMessage(
                content=[
                    {"type": "text", "text": final_prompt_text},
                    {"type": "image_url", "image_url": image_to_process}
                ]
            )
            # Gọi invoke với một danh sách message
            response = llm.invoke([message])
            # Kết quả trả về là một AIMessage, ta cần lấy .content
            answer = response.content
        else:
            # **Trường hợp chỉ có text**
            # Gọi invoke trực tiếp với chuỗi string
            response = llm.invoke(final_prompt_text)
            # Kết quả trả về là một chuỗi string
            answer = response

        print("Đã nhận được câu trả lời.")

        sources = [{"source": os.path.basename(doc.metadata.get('source', 'N/A')), "content": doc.page_content} for doc
                   in retrieved_docs]

        return jsonify({"answer": answer, "sources": sources})
    except Exception as e:
        print(f"Lỗi khi trả lời câu hỏi: {e}")
        return jsonify({"error": str(e)}), 500


# --- KHỞI ĐỘNG SERVER ---
if __name__ == '__main__':
    print("--- Backend Server sẵn sàng lắng nghe trên cổng 5001 ---")
    app.run(debug=True, port=5001)