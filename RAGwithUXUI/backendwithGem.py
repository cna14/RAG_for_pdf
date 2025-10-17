from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
import fitz
from PIL import Image
import pytesseract
import base64
import io

# LangChain components
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from langchain.retrievers.multi_query import MultiQueryRetriever

# Import thư viện gốc của Google và các loại dữ liệu của nó
import google.generativeai as genai
from google.generativeai.types import GenerateContentResponse

# --- CẤU HÌNH & KHỞI TẠO ---
print("--- Khởi tạo Backend Server ---")
load_dotenv()

try:
    pytesseract.pytesseract.tesseract_cmd = r'E:\university\V.1\chuyendeCNTT\Tesseract\tesseract.exe'
    print("Đã tìm thấy Tesseract OCR.")
except Exception:
    print("CẢNH BÁO: Không tìm thấy Tesseract tại đường dẫn đã chỉ định.")

VECTOR_STORES_MAIN_DIR = "vector_stores"
PDF_UPLOADS_DIR = "pdf_uploads"
os.makedirs(VECTOR_STORES_MAIN_DIR, exist_ok=True)
os.makedirs(PDF_UPLOADS_DIR, exist_ok=True)

app = Flask(__name__)
CORS(app)

print("Đang tải các mô hình AI...")
try:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    llm_for_retriever = GoogleGenerativeAI(model="gemini-2.0-flash-lite-001", google_api_key=os.getenv("GOOGLE_API_KEY"))
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    native_gemini_model = genai.GenerativeModel("gemini-2.0-flash-lite-001")
    print("Các mô hình AI đã sẵn sàng.")
except Exception as e:
    print(f"LỖI NGHIÊM TRỌNG: Không thể tải mô hình AI. Lỗi: {e}")
    native_gemini_model = None


# --- CÁC HÀM LOGIC ---

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


def analyze_attached_image(image_object):
    """Thực hiện cả OCR và phân tích trực quan trên ảnh đính kèm."""
    ocr_text = ""
    visual_description = ""
    try:
        ocr_text = pytesseract.image_to_string(image_object, lang='vie+eng').strip()
        if ocr_text: print(f"Văn bản OCR từ ảnh: '{ocr_text}'")
    except Exception as e:
        print(f"Lỗi khi OCR ảnh đính kèm: {e}")
    try:
        print("Đang gửi ảnh đến Gemini để phân tích trực quan...")
        prompt = "Mô tả chi tiết các yếu tố trực quan trong ảnh (biểu đồ, bố cục, đối tượng). Bỏ qua việc trích xuất lại văn bản thô."
        response = native_gemini_model.generate_content([prompt, image_object])
        visual_description = response.text.strip()
        if visual_description: print(f"Mô tả trực quan: '{visual_description}'")
    except Exception as e:
        print(f"Lỗi khi phân tích trực quan ảnh: {e}")
        visual_description = "Không thể phân tích các yếu tố trực quan của hình ảnh."
    return ocr_text, visual_description


# --- CÁC ĐIỂM CUỐI API (ENDPOINTS) ---

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
    if native_gemini_model is None:
        return jsonify({"error": "Mô hình AI chưa được tải."}), 503

    data = request.json
    original_question = data.get('question', '')
    chat_id = data.get('chatId')
    image_data_url = data.get('image')

    if not chat_id:
        return jsonify({"error": "Thiếu ID của dự án (chatId)."}), 400

    text_from_image_ocr = ""
    visual_description = ""
    image_to_process = None

    if image_data_url:
        try:
            header, encoded = image_data_url.split(",", 1)
            image_to_process = Image.open(io.BytesIO(base64.b64decode(encoded)))
            text_from_image_ocr, visual_description = analyze_attached_image(image_to_process)
        except Exception as e:
            return jsonify({"error": f"Lỗi xử lý ảnh đính kèm: {e}"}), 400

    main_task = original_question if original_question else text_from_image_ocr if text_from_image_ocr else "Hãy phân tích hình ảnh này và liên hệ với ngữ cảnh tài liệu nếu có."
    print(f"Nhiệm vụ chính được xác định là: '{main_task}'")

    vector_store_path = os.path.join(VECTOR_STORES_MAIN_DIR, chat_id)
    retrieved_docs = []
    if os.path.exists(vector_store_path) and main_task:
        try:
            vectorstore = Chroma(persist_directory=vector_store_path, embedding_function=embeddings)
            retriever = MultiQueryRetriever.from_llm(retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
                                                     llm=llm_for_retriever)
            retrieved_docs = retriever.invoke(main_task)
        except Exception as e:
            print(f"Lỗi khi truy xuất tài liệu: {e}")

    context_text = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])

    try:
        prompt_template = """Bạn là một trợ lý AI chuyên gia, có khả năng phân tích đa phương thức.
        Mục tiêu của bạn là hoàn thành **NHIỆM VỤ CHÍNH** dưới đây bằng cách sử dụng các nguồn thông tin được cung cấp.

        **NHIỆM VỤ CHÍNH (MAIN TASK):**
        {main_task}

        ---
        **CÁC NGUỒN THÔNG TIN HỖ TRỢ:**
        1.  `NGỮ CẢNH TỪ TÀI LIỆU`: Các đoạn văn bản đã được trích xuất từ file PDF có liên quan đến nhiệm vụ.
            {context}
        2.  `PHÂN TÍCH HÌNH ẢNH ĐÍNH KÈM`:
            - Văn bản đọc được từ ảnh (OCR): {image_ocr_text}
            - Mô tả các yếu tố trực quan (biểu đồ, bố cục): {image_visual_description}
        3.  `CÂU HỎI GỐC CỦA NGƯỜI DÙNG` (nếu có): {original_question}

        **QUY TRÌNH TƯ DUY BẮT BUỘC:**
        1.  **Tập trung vào Nhiệm vụ chính:** Hiểu rõ yêu cầu trong `NHIỆM VỤ CHÍNH`.
        2.  **Tìm kiếm bằng chứng:** Tìm kiếm câu trả lời cho `NHIỆM VỤ CHÍNH` trong `NGỮ CẢNH TỪ TÀI LIỆU` trước tiên.
        3.  **Đối chiếu và Mở rộng:** Sử dụng `PHÂN TÍCH HÌNH ẢNH ĐÍNH KÈM` như những bằng chứng bổ sung hoặc nguồn thông tin chính nếu `NGỮ CẢNH TỪ TÀI LIỆU` không đủ.
        4.  **Hành động như chuyên gia:** Đưa ra câu trả lời được suy luận, tổng hợp, không chỉ sao chép.
        5.  **Bám sát sự thật và định dạng đẹp:** (Giữ nguyên các quy tắc cũ của bạn).

        ---
        **CÂU TRẢ LỜI CỦA CHUYÊN GIA (tập trung vào việc hoàn thành NHIỆM VỤ CHÍNH):**"""

        final_prompt_text = prompt_template.format(
            main_task=main_task,
            context=context_text,
            image_ocr_text=text_from_image_ocr,
            image_visual_description=visual_description,
            original_question=original_question
        )

        model_input = [final_prompt_text]
        if image_to_process:
            model_input.append(image_to_process)

        print("Đang gửi yêu cầu (trực tiếp, đã tiền xử lý OCR và Vision) đến mô hình Gemini...")
        response = native_gemini_model.generate_content(model_input)

        ### GIẢI PHÁP TRIỆT ĐỂ: KẾT HỢP LẬP TRÌNH PHÒNG THỦ ###
        # ==============================================================================
        answer = ""
        # Trường hợp 1: Là đối tượng GenerateContentResponse (chuẩn của thư viện gốc)
        if isinstance(response, GenerateContentResponse):
            answer = response.text
        # Trường hợp 2: Bị can thiệp, trả về đối tượng có thuộc tính .content
        elif hasattr(response, 'content'):
            answer = response.content
        # Trường hợp 3: Bị can thiệp, trả về một chuỗi string đơn thuần
        elif isinstance(response, str):
            answer = response
        # Trường hợp không xác định
        else:
            raise TypeError(f"Không thể xử lý định dạng trả về không xác định: {type(response)}")
        # ==============================================================================

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