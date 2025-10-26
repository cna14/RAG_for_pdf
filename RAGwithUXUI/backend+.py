from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import base64
import io
import datetime  # Import thư viện datetime

# LangChain components (CHỈ DÙNG CHO RAG)
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

# Cấu hình đường dẫn Tesseract (QUAN TRỌNG: Phải chính xác)
try:
    pytesseract.pytesseract.tesseract_cmd = r'E:\university\V.1\chuyendeCNTT\Tesseract\tesseract.exe'
    print("Đã tìm thấy Tesseract OCR.")
except Exception:
    print("CẢNH BÁO: Không tìm thấy Tesseract tại đường dẫn đã chỉ định.")

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
    # 1. Mô hình Embedding (của LangChain, ổn định và miễn phí)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 2. Mô hình LLM của LangChain (CHỈ DÙNG CHO RETRIEVER)
    # Sử dụng model Pro để có MultiQuery tốt nhất
    llm_for_retriever = GoogleGenerativeAI(model="gemini-2.0-flash-lite-001", google_api_key=os.getenv("GOOGLE_API_KEY"))

    # 3. Cấu hình và tải mô hình GỐC của Google (DÙNG ĐỂ TẠO CÂU TRẢ LỜI)
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    # Sử dụng model Pro để có khả năng Vision và suy luận tốt nhất
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
    """Học file PDF, thêm timestamp vào metadata, và lưu vào DB."""
    document_text = extract_text_from_pdf_with_ocr(pdf_path)
    if not document_text.strip():
        raise ValueError("Không trích xuất được văn bản nào từ file PDF.")
    print("Đang xây dựng cơ sở tri thức...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)

    current_time_iso = datetime.datetime.now().isoformat()
    base_metadata = {"source": pdf_path, "learned_at": current_time_iso}
    metadatas = [base_metadata.copy() for _ in text_splitter.split_text(document_text)]
    chunks = text_splitter.create_documents([document_text], metadatas=metadatas)

    db = Chroma.from_documents(chunks, embeddings, persist_directory=vector_store_path)
    db.persist()
    print(f"Xây dựng cơ sở tri thức thành công lúc: {current_time_iso}")


def analyze_attached_image(image_object):
    """Thực hiện cả OCR và phân tích trực quan trên ảnh đính kèm."""
    ocr_text = ""
    visual_description = ""
    # 1. Thực hiện OCR
    try:
        ocr_text = pytesseract.image_to_string(image_object, lang='vie+eng').strip()
        if ocr_text: print(f"Văn bản OCR từ ảnh: '{ocr_text}'")
    except Exception as e:
        print(f"Lỗi khi OCR ảnh đính kèm: {e}")
    # 2. Thực hiện phân tích trực quan bằng Gemini
    try:
        if native_gemini_model:
            print("Đang gửi ảnh đến Gemini để phân tích trực quan...")
            # Sử dụng model hỗ trợ vision (native_gemini_model đã là Pro)
            prompt_vision = "Mô tả chi tiết các yếu tố trực quan trong ảnh (biểu đồ, bố cục, đối tượng). Bỏ qua việc trích xuất lại văn bản thô."
            # Quan trọng: Gửi ảnh theo cách thư viện gốc yêu cầu
            response_vision = native_gemini_model.generate_content([prompt_vision, image_object])
            visual_description = response_vision.text.strip()
            if visual_description: print(f"Mô tả trực quan: '{visual_description}'")
        else:
            print("CẢNH BÁO: Không có mô hình AI để phân tích trực quan.")
            visual_description = "Không thể phân tích trực quan do lỗi tải mô hình."
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
    retrieved_docs_initial = []  # Kết quả tìm kiếm ban đầu
    final_docs_for_context = []  # Kết quả sau khi sắp xếp và chọn lọc

    if os.path.exists(vector_store_path) and main_task:
        try:
            vectorstore = Chroma(persist_directory=vector_store_path, embedding_function=embeddings)
            # Truy xuất rộng hơn (k=10)
            retriever = MultiQueryRetriever.from_llm(retriever=vectorstore.as_retriever(search_kwargs={"k": 10}),
                                                     llm=llm_for_retriever)
            retrieved_docs_initial = retriever.invoke(main_task)

            # Sắp xếp lại theo thời gian giảm dần
            def get_timestamp(doc):
                ts_str = doc.metadata.get('learned_at', datetime.datetime.min.isoformat())
                try:
                    return datetime.datetime.fromisoformat(ts_str)
                except ValueError:
                    return datetime.datetime.min

            sorted_docs = sorted(retrieved_docs_initial, key=get_timestamp, reverse=True)

            # Chọn top 5 tài liệu mới nhất
            final_docs_for_context = sorted_docs[:5]

        except Exception as e:
            print(f"Lỗi khi truy xuất tài liệu: {e}")

    # Tạo context từ các tài liệu đã được chọn lọc và sắp xếp
    context_parts = []
    for doc in final_docs_for_context:
        learn_time_str = doc.metadata.get('learned_at', 'Không rõ')
        try:
            # Định dạng lại timestamp cho dễ đọc hơn (tùy chọn)
            learn_time = datetime.datetime.fromisoformat(learn_time_str).strftime('%Y-%m-%d %H:%M')
        except:
            learn_time = learn_time_str  # Giữ nguyên nếu không parse được
        context_parts.append(
            f"[Nguồn: {os.path.basename(doc.metadata.get('source', 'N/A'))} | Học lúc: {learn_time}]\n{doc.page_content}")
    context_text = "\n\n---\n\n".join(context_parts)

    try:
        # PROMPT ĐÃ ĐƯỢC NÂNG CẤP ĐỂ ƯU TIÊN THÔNG TIN MỚI
        prompt_template = """Bạn là một trợ lý AI chuyên gia, có khả năng phân tích đa phương thức.
        Mục tiêu của bạn là hoàn thành **NHIỆM VỤ CHÍNH** dưới đây bằng cách sử dụng các nguồn thông tin được cung cấp.

        **NHIỆM VỤ CHÍNH (MAIN TASK):**
        {main_task}

        ---
        **CÁC NGUỒN THÔNG TIN HỖ TRỢ:**
        1.  `NGỮ CẢNH TỪ TÀI LIỆU (Đã ưu tiên thông tin mới)`: Các đoạn văn bản từ PDF, sắp xếp theo thời gian học gần đây nhất lên đầu.
            {context}
        2.  `PHÂN TÍCH HÌNH ẢNH ĐÍNH KÈM` (nếu có):
            - Văn bản đọc được từ ảnh (OCR): {image_ocr_text}
            - Mô tả các yếu tố trực quan (biểu đồ, bố cục): {image_visual_description}
        3.  `CÂU HỎI GỐC CỦA NGƯỜI DÙNG` (nếu có): {original_question}

        **QUY TRÌNH TƯ DUY BẮT BUỘC:**
        1.  **Tập trung vào Nhiệm vụ chính.**
        2.  **Tìm kiếm bằng chứng** trong `NGỮ CẢNH TỪ TÀI LIỆU`.
        3.  **ƯU TIÊN THÔNG TIN MỚI:** Ngữ cảnh đã được sắp xếp ưu tiên thông tin mới nhất lên đầu (dựa vào [Học lúc: ...]). Hãy **đặc biệt chú trọng** những thông tin này khi tổng hợp câu trả lời, trừ khi nhiệm vụ yêu cầu thông tin lịch sử.
        4.  **Đối chiếu và Mở rộng** với `PHÂN TÍCH HÌNH ẢNH ĐÍNH KÈM`.
        5.  **Hành động như chuyên gia:** Đưa ra câu trả lời được suy luận, tổng hợp.
        6.  **Bám sát sự thật và định dạng đẹp:** (Giữ nguyên các quy tắc cũ của bạn).

        ---
        **CÂU TRẢ LỜI CỦA CHUYÊN GIA (ưu tiên thông tin mới nhất nếu phù hợp):**"""

        final_prompt_text = prompt_template.format(
            main_task=main_task,
            context=context_text,
            image_ocr_text=text_from_image_ocr,
            image_visual_description=visual_description,
            original_question=original_question
        )

        # Chuẩn bị đầu vào cho thư viện gốc của Google
        model_input = [final_prompt_text]
        if image_to_process:
            model_input.append(image_to_process)

        print("Đang gửi yêu cầu (trực tiếp, đã tiền xử lý) đến mô hình Gemini...")
        response = native_gemini_model.generate_content(model_input)

        # GIẢI PHÁP TRIỆT ĐỂ: KẾT HỢP LẬP TRÌNH PHÒNG THỦ
        # ==============================================================================
        answer = ""
        # Trường hợp 1: Là đối tượng GenerateContentResponse (chuẩn của thư viện gốc)
        if isinstance(response, GenerateContentResponse):
            try:
                answer = response.text
            except Exception as e:
                print(f"Lỗi khi truy cập response.text: {e}. Có thể do safety settings.")
                # Cố gắng lấy từ parts nếu có
                try:
                    answer = response.parts[0].text if response.parts else "Phản hồi bị chặn hoặc không có nội dung."
                except Exception:
                    answer = "Lỗi không xác định khi xử lý phản hồi."

        # Trường hợp 2: Bị can thiệp (ít khả năng xảy ra với cách gọi trực tiếp)
        elif hasattr(response, 'content'):
            answer = response.content
        elif isinstance(response, str):
            answer = response
        else:
            print(f"Định dạng trả về không xác định: {type(response)}")
            answer = "Lỗi: Định dạng phản hồi không mong đợi."
        # ==============================================================================

        print("Đã nhận được câu trả lời.")

        # Trả về nguồn là các tài liệu đã được dùng làm ngữ cảnh cuối cùng
        sources = [{"source": os.path.basename(doc.metadata.get('source', 'N/A')),
                    "content": doc.page_content,
                    "metadata": doc.metadata}
                   for doc in final_docs_for_context]

        return jsonify({"answer": answer, "sources": sources})
    except Exception as e:
        print(f"Lỗi khi trả lời câu hỏi: {e}")
        # Cố gắng trả về thông báo lỗi cụ thể hơn nếu có
        error_message = str(e)
        if "API key not valid" in error_message:
            error_message = "Lỗi xác thực: API Key không hợp lệ hoặc đã hết hạn."
        elif "quota" in error_message.lower():
            error_message = "Lỗi hạn mức: Đã vượt quá số lượt gọi API cho phép. Vui lòng thử lại sau hoặc kiểm tra gói cước."
        elif "Content has no parts" in error_message:
            error_message = "Lỗi nội dung: Phản hồi từ AI bị chặn do cài đặt an toàn."

        return jsonify({"error": error_message}), 500


# --- KHỞI ĐỘNG SERVER ---
if __name__ == '__main__':
    print("--- Backend Server sẵn sàng lắng nghe trên cổng 5001 ---")
    app.run(debug=True, port=5001)